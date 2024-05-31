import os
import re
from word2number import w2n
import copy
import asyncio
from typing import List, Tuple, Optional, Dict, Union, Any, NoReturn
from omegaconf import DictConfig
from hydra import compose
from hydra.utils import instantiate
import traceback
from sqlalchemy import desc, select, delete, update

from sqlalchemy.ext.asyncio import AsyncSession
from openai.types.beta.threads import (
    TextContentBlock,
    ImageFileContentBlock,
    ImageURLContentBlock,
)

from enum import Enum


import logging
import nest_asyncio

from fastapi import WebSocket

nest_asyncio.apply()

from services.openai_assistants.chat_assistant.chat_assistant import ChatAssistant
from services.message_sender import MessageSender
from services.openai_assistants.helper_assistant.helper_assistant import HelperAssistant
from services.utils.file_utils import (
    create_new_story_directory,
    get_latest_story_directory,
    convert_user_info_to_json_files,
    convert_user_info_to_md_files,
)
from services.chapter_generator.chapter_generation_mechanism import (
    ChapterGenerationMechanism,
)
from services.doc_generator.doc import StoryDocument
from services.image_prompt_generator.image_prompt_generation_mechanism import (
    ImagePromptGenerationMechanism,
)
from services.image_generators.dalle3 import DALLE3ImageGenerator
from services.utils.log_utils import get_logger
from services.session_service import refresh_access_token

from magic_tales_models.models.story import Story
from magic_tales_models.models.story_state import StoryState

from .story_manager import StoryManager
from .database_manager import DatabaseManager
from magic_tales_models.models.command import Command
from magic_tales_models.models.response import ResponseStatus
from magic_tales_models.models.message import (
    MessageSchema,
    OriginEnum,
    TypeEnum,
)
from magic_tales_models.models.profile import Profile
from magic_tales_models.models.user import User
from magic_tales_models.models.ws_input import WSInput
from magic_tales_models.models.ws_output import WSOutput


# Set up logging
logging.basicConfig(level=logging.INFO)
# Get a logger instance for this module
logger = get_logger(__name__)


os.environ["NUMEXPR_MAX_THREADS"] = "16"  # Set to the number of cores you wish to use


class MagicTalesCoreOrchestrator:
    """
    Core orchestrator for Magic Tales application. Manages the story generation process.
    """

    steps_execution_mapping: Dict[StoryState, str] = {
        StoryState.STORY_GENERATION: "_generate_story",
        StoryState.IMAGE_PROMPT_GENERATION: "_generate_image_prompts",
        StoryState.IMAGE_GENERATION: "_generate_images",
        StoryState.DOCUMENT_GENERATION: "_generate_final_document",
    }

    def __init__(
        self,
        config: DictConfig,
        message_sender: MessageSender,
        session: AsyncSession,
        websocket: WebSocket,
    ) -> None:
        """
        Initialize the Magic Tales Core server.

        Args:
            config (DictConfig): Configuration parameters.
        """
        # Initialize logging
        logger.info("Initializing MagicTales.")

        self._validate_openai_api_key()

        # Configuration
        self.config = copy.deepcopy(config)

        # Initialize the Helper Assistant
        self.helper_assistant = HelperAssistant(config.helper_assistant)

        # Initialize the Chat Assistant
        self.chat_assistant = ChatAssistant(config=self.config.chat_assistant)

        # Subfolders for storing various assets
        self.subfolders = {"images": "images", "chapters": "chapters"}

        self.chapter_generation_mechanism = None
        self.image_prompt_generation_mechanism = None

        self.message_sender = message_sender
        self.session = session
        self.websocket = websocket
        self.token_data = None

        self.user_id = None
        self.user_dict = {}
        self.latest_story = None

        self.new_token = None

        self.token_refresh_interval = 120  # 2 minutes in seconds
        self.token_refresh_task = None

        self.database_manager = DatabaseManager(session=session)

        self.story_manager = StoryManager(db_manager=self.database_manager)

        self.wait_for_response_to_continue_where_we_left_off = False

        self.frontend_command_handlers = {
            Command.NEW_TALE: self.handle_command_new_tale,
            Command.SPIN_OFF: self.handle_command_spin_off,
            Command.UPDATE_PROFILE: self.handle_command_update_profile,
            Command.CONVERSATION_RECOVERY: self.handle_command_conversation_recovery,
            Command.LINK_USER_WITH_CONVERSATIONS: self.handle_command_link_user_with_conversations,
            Command.USER_MESSAGE: self._handle_user_message,
            Command.USER_LOGGED_IN: self.last_story_finished_correctly,
        }

        logger.info("MagicTales initialized.")

    def _validate_openai_api_key(self) -> None:
        """Validates the presence of the OpenAI API key."""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        
    def _cancel_token_refresh_task(self):
        if self.token_refresh_task:
            self.token_refresh_task.cancel()

    async def process_frontend_request(
        self, frontend_request: WSInput, token_data: dict
    ) -> None:
        """
        Process incoming commands/requests from the AI Core Interface Layer.

        Args:
            frontend_request (WSInput): Full command structure.
            token_data (dict): Token data from the AI Core Interface Layer.

        Returns:
            None
        """
        self.token_data = token_data
        if not self.token_refresh_task or self.token_refresh_task.done():
            self.new_token = await refresh_access_token(frontend_request.token)
            self.token_refresh_task = asyncio.create_task(
                self.refresh_token_periodically()
            )

        coming_user_id = token_data.get("user_id", None)

        if frontend_request.command in [
            Command.NEW_TALE,
            Command.SPIN_OFF,
            Command.UPDATE_PROFILE,
            Command.LINK_USER_WITH_CONVERSATIONS,
        ]:
            await self.send_message_to_frontend(
                WSOutput(
                    command=frontend_request.command, token=self.new_token, ack=True
                )
            )

        await self.send_working_command_to_frontend(True)

        if self.user_id != coming_user_id:
            await self._handle_user_id_change(coming_user_id, frontend_request.try_mode)

        await self._save_command_message(frontend_request)

        await self.process_command(frontend_request)

        logger.info(f"Processed command: {frontend_request.command}")
        await self.send_working_command_to_frontend(False)

    async def _handle_user_id_change(self, coming_user_id: int, try_mode: bool) -> None:
        """
        Handle the change in user ID and initialize user-specific resources.

        Args:
            coming_user_id (int): The new user ID.
        """
        self.user_id = coming_user_id
        self.user_dict = (
            await self.database_manager.fetch_user_by_id(self.user_id)
        ).to_dict()
        if not self.user_dict:
            logger.warn(f"User {self.user_id} has not been found.")
            raise ValueError(f"User {self.user_id} has not been found.")

        assistant_id = self.user_dict.get("assistant_id", None)
        helper_id = self.user_dict.get("helper_id", None)

        logger.info(f"User current Assistant id: {assistant_id}")
        logger.info(f"User current Helper id: {helper_id}")

        files_paths = await self._fetch_user_data_and_update_knowledge_base()
        await self._initialize_assistants(assistant_id, helper_id, files_paths)

        user_message = self._generate_starting_message(try_mode=try_mode)
        await self._generate_and_send_update_message_for_user(user_message)

        await self.story_manager.refresh()

    async def process_command(self, frontend_request: WSInput):
        """
        Process the incoming command using the appropriate handler from the self.frontend_command_handlers dictionary.

        Args:
            frontend_request (WSInput): The request from the frontend containing the command to process.
        """
        handler = self.frontend_command_handlers.get(frontend_request.command)
        if handler:
            await handler(frontend_request)
        else:
            logger.warning(f"Unknown command: {frontend_request.command}")

    async def _initialize_assistants(
        self,
        assistant_id: Optional[str],
        helper_id: Optional[str],
        files_paths: Optional[List[str]],
    ) -> None:
        """
        Initialize chat and helper assistants.

        Args:
            assistant_id (Optional[str]): The ID of the chat assistant.
            helper_id (Optional[str]): The ID of the helper assistant.
        """
        try:
            await self.chat_assistant.start_assistant(assistant_id, files_paths)
        except ValueError as e:
            logger.error(f"Error initializing chat assistant: {e}")
            await self._send_error_message(
                "We are currently experiencing technical difficulties. Please try again later."
            )
            return

        if assistant_id != self.chat_assistant.openai_assistant.id:
            logger.warn(
                f"User {self.user_id} had CHAT assistant {assistant_id}, which was not found. It's going to be updated. This should only happen with brand new users!"
            )
            await self.database_manager.update_user_record_by_id(
                self.user_id, {"assistant_id": self.chat_assistant.openai_assistant.id}
            )

        try:
            await self.helper_assistant.start_assistant(helper_id)
        except ValueError as e:
            logger.error(f"Error initializing helper assistant: {e}")
            await self._send_error_message(
                "We are currently experiencing technical difficulties. Please try again later."
            )
            return

        if helper_id != self.helper_assistant.openai_assistant.id:
            logger.warn(
                f"User {self.user_id} had HELPER assistant {helper_id}, which was not found. It's going to be updated. This should only happen with brand new users!"
            )
            await self.database_manager.update_user_record_by_id(
                self.user_id, {"helper_id": self.helper_assistant.openai_assistant.id}
            )

    async def handle_command_new_tale(self, frontend_request: WSInput):
        if await self.last_story_finished_correctly(self.user_id):
            asyncio.create_task(self._handle_new_tale())
        else:
            self.wait_for_response_to_continue_where_we_left_off = True

    async def handle_command_spin_off(self, frontend_request: WSInput):
        if not frontend_request.story_id:
            raise Exception("story_id is required for spin-off")
        asyncio.create_task(self._handle_spin_off_tale(frontend_request))

    async def handle_command_update_profile(self, frontend_request: WSInput):
        if not frontend_request.profile_id:
            raise Exception("profile_id is required for update profile")
        asyncio.create_task(self._handle_user_request_update_profile(frontend_request))

    async def handle_command_conversation_recovery(self, frontend_request: WSInput):
        if not self.websocket.uid:
            raise Exception("uid is required for conversation recovery")
        conversations = await self.database_manager.fetch_messages_for_session(
            self.websocket.uid
        )
        conversation_dicts = [
            MessageSchema().dump(conversation) for conversation in conversations
        ]
        await self.send_message_to_frontend(
            WSOutput(
                command=frontend_request.command,
                token=self.new_token,
                data={"conversations": conversation_dicts},
            )
        )

    async def handle_command_link_user_with_conversations(
        self, frontend_request: WSInput
    ):
        if not self.user_id:
            raise Exception("user_id is required for link user with conversations")
        if not frontend_request.session_ids:
            raise Exception("session_ids are required for link user with conversations")
        await self.database_manager.link_user_with_conversations_by_session_ids(
            frontend_request.session_ids
        )
        await self.story_manager.refresh()

    async def _save_command_message(self, frontend_request: WSInput) -> None:
        """
        Save the incoming command message to the database.

        Args:
            frontend_request (WSInput): The incoming command request.
        """
        msg_type = (
            TypeEnum.chat
            if frontend_request.command == Command.USER_MESSAGE
            else TypeEnum.command
        )
        await self.database_manager.add_message(
            self.user_id,
            self.websocket.uid,
            frontend_request.command,
            OriginEnum.user,
            msg_type,
            frontend_request.model_dump(),
        )
        await self.story_manager.refresh()

    async def _handle_conversation_recovery(self, frontend_request: WSInput) -> None:
        """
        Handle the conversation recovery command.

        Args:
            frontend_request (WSInput): The incoming command request.
        """
        if not self.websocket.uid:
            raise ValueError("uid is required for conversation recovery")

        conversations = await self.database_manager.fetch_messages_for_session(
            self.websocket.uid
        )
        conversation_dicts = [
            MessageSchema().dump(conversation) for conversation in conversations
        ]

        await self.send_message_to_frontend(
            WSOutput(
                command=frontend_request.command,
                token=self.new_token,
                data={"conversations": conversation_dicts},
            )
        )

    async def _handle_link_user_with_conversations(
        self, frontend_request: WSInput
    ) -> None:
        """
        Handle the link user with conversations command.

        Args:
            frontend_request (WSInput): The incoming command request.
        """
        if not self.user_id:
            raise ValueError("user_id is required for link user with conversations")

        if not frontend_request.session_ids:
            raise ValueError(
                "session_ids are required for link user with conversations"
            )

        await self.database_manager.link_user_with_conversations_by_session_ids(
            frontend_request.session_ids
        )
        await self.story_manager.refresh()

    async def _send_error_message(self, message: str) -> None:
        """
        Send an error message to the frontend.

        Args:
            message (str): The error message to send.
        """
        await self.send_message_to_frontend(
            WSOutput(
                command=Command.MESSAGE_FOR_HUMAN, token=self.new_token, message=message
            )
        )

    async def fetch_latest_story_for_user(self, user_id):
        """
        Fetches the latest story that the user interacted with by leveraging the DatabaseManager to reduce direct database calls.

        Args:
            user_id (int): The user ID whose latest story is being fetched.

        Returns:
            Story or None: The most recently updated story if available, otherwise None.

        Raises:
            Exception: If there is an error during the database operation or processing.
        """
        try:
            # Fetch all stories for the user using the DatabaseManager
            _, stories = (
                await self.database_manager.fetch_profiles_and_stories_for_user(
                    user_id=user_id
                )
            )

            # Determine the latest story based on the 'last_updated' field
            latest_story = max(
                stories, key=lambda story: story.last_updated, default=None
            )

            if latest_story:
                logger.info(
                    f"Latest story for user {user_id} is story ID {latest_story.id} from profile ID {latest_story.profile_id}"
                )
            else:
                logger.info(f"No stories found for user {user_id}.")

            return latest_story

        except Exception as e:
            logger.error(f"Error fetching the last story for user {user_id}: {str(e)}")
            return None

    async def last_story_finished_correctly(self, user_id) -> bool:
        self.latest_story = await self.fetch_latest_story_for_user(user_id)
        if self.latest_story:
            if (
                self.latest_story.last_successful_step
                == StoryState.FINAL_DOCUMENT_GENERATED.value
            ):
                # If the last story is completed, log this event and wait for further user actions.
                logger.info(
                    "Last story completed; waiting for user to start a new story."
                )
                return True
            else:
                logger.info(
                    f"Last story completed step was: {StoryState(self.latest_story.last_successful_step).name}"
                )
                # Process to resume the story
                await self._generate_and_send_update_message_for_user(
                    self.config.updates_request_prompts.incomplete_story_found
                )
                return False
        else:
            # Log that no story is available; await user action to start new.
            logger.info(
                "No story available for user; awaiting initiation of new story."
            )
            return False

    async def _resume_story(self):
        """
        Resumes the story generation process from the last saved state, ensuring that the database session is correctly managed.
        """
        logger.info("Resuming story generation from the last saved state.")

        # Check if the session is still open and valid; if not, manage it here or ensure it's managed externally
        if not self.session:
            logger.error(
                "Session is closed or invalid when trying to resume the story."
            )
            return

        if not self.latest_story:
            logger.error("Last story that we are supposed to recover from, is None!")
            return

        try:
            # Load story using the current session
            await self.session.refresh(self.latest_story)
            await self.story_manager.load_story(self.latest_story.id)
            self.latest_story = None

            if self.story_manager.story:
                last_step = await self.story_manager.get_last_step()
                logger.info(f"Resuming story generation from step: {str(last_step)}")
                await self._process_story_generation_step(last_step)
            else:
                logger.info("No saved state found, awaiting user action.")
                # Clean up if no story data was found
                self.story_manager.reset()

        except Exception as e:
            logger.error(f"Failed to resume story due to: {e}")
            # Handle session rollback if needed
            await self.story_manager.session.rollback()

    async def send_working_command_to_frontend(self, is_working: bool):
        """
        Sends a message to the frontend if the AI start working or stop working.
        If 'is_working' variable is equals to True, it indicate that the AI started working, and if it's False, it indicates that the AI finished the process.
        """
        # Send working command to the frontend
        if is_working:
            command = Command.IS_WORKING
        else:
            command = Command.DONE_WORKING

        await self.message_sender.send_message_to_frontend(
            WSOutput(command=command, token=self.new_token)
        )

    async def send_message_to_frontend(self, output_model: WSOutput):
        """
        Sends a message to the AI Core Interface Layer to be sent to the Front End.
        """
        await self.message_sender.send_message_to_frontend(output_model)

        # Now we need to save the message to the database
        details_serialized = output_model.dict()
        # Iterate over all keys in the serialized details and check for Enums
        for key, value in details_serialized.items():
            if isinstance(value, Enum):
                details_serialized[key] = value.value
            elif value is None:
                # Optionally remove None values if that's desired
                details_serialized[key] = (
                    None  # or use `del details_serialized[key]` if removing
                )

        # Now `details_serialized` contains only JSON-serializable types
        await self.database_manager.add_message(
            self.user_id,
            self.websocket.uid,
            output_model.command,
            OriginEnum.ai,
            TypeEnum.chat,
            details_serialized,
        )
        await self.story_manager.refresh()

    async def create_and_send_progress_update(self, **kwargs) -> None:
        """
        Creates a progress update message in JSON format.

        This method dynamically constructs a progress update message based on the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments representing various components of the progress update message.

        """
        progress_update = WSOutput(
            command=Command.PROGRESS_UPDATE,
            token=self.new_token,
        )

        # Iterate through the keyword arguments and set them on the progress_update instance
        for key, value in kwargs.items():
            if hasattr(progress_update, key):
                setattr(progress_update, key, value)
            else:
                raise KeyError(f"Invalid progress update parameter: '{key}'")

        await self.send_message_to_frontend(progress_update)
        return

    async def _handle_user_message(self, request: WSInput) -> None:
        """
        Handle a user message by sending it to the chat assistant and processing the response.

        Args:
            user_message (str): The user's message to be processed.
            websocket (str): Unique identifier of the client.

        Returns:
            None
        """
        user_message = request.message

        # Generate AI response for the user message
        ai_message_for_human, ai_message_for_system = (
            await self.chat_assistant.request_ai_response(message=user_message)
        )

        logger.info(f"AI response for human/user: {ai_message_for_human}")

        # Construct a response to be sent by the AI Core Interface Layer to the Front End
        ai_response = WSOutput(
            command=Command.MESSAGE_FOR_HUMAN,
            message=ai_message_for_human,  # AI-generated response to be displayed to the user
            token=self.new_token,
            working=False,  # Indicates that the response is ready
        )
        await self.send_message_to_frontend(ai_response)

        if ai_message_for_system:
            logger.warn(
                f"AI Sent a COMMAND for the Orchestrator: {ai_message_for_system}"
            )
            asyncio.create_task(self.handle_assistant_requests(ai_message_for_system))

    async def _handle_user_request_update_profile(self, request: WSInput) -> None:
        """
        Handle the 'user_request_update_profile' command by retrieving user information from the database, creating a chat where the user might night give us new information about this profile.

        Returns:
            None.
        """
        logger.info("Updating profile.")

        # await self._reset()

    async def _handle_spin_off_tale(self, request: WSInput) -> None:
        """
        Handle the 'spin_off' command by retrieving user information from the database.

        Returns:
            None.
        """
        logger.info("Starting a spin-off story generation.")

        await self._reset()

    async def _handle_new_tale(self) -> None:
        """
        Handle the 'new-tale' command by retrieving user information from the database.

        Returns:
            None.
        """
        logger.info("Starting story generation from scratch.")
        await self._reset()
        await self._generate_and_send_update_message_for_user(
            self.config.updates_request_prompts.new_tale_clicked
        )

    async def _generate_and_send_update_message_for_user(
        self, request_message: str
    ) -> None:
        """
        Sends an update message to the user.

        Args:
            request_message (str): This is the message we are sending the Assistant to generate an amazing update for the user.

        Returns:
            None.
        """
        request_message = (
            request_message.replace("{user_info}", f"{self.user_dict}")
            + " Use strictly the same language that is being used in the conversation."
        )
        request_message = WSInput(
            command=Command.USER_MESSAGE, token=self.new_token, message=request_message
        )
        # asyncio.create_task(self._handle_user_message(request_message))
        await self._handle_user_message(request_message)

    async def _fetch_user_data_and_update_knowledge_base(self) -> List[str]:
        """
        Retrieves user data from the database and creates a knowledge base file.

        Returns:
            List[str]: List of file paths to the created knowledge base files.
        """
        if self.user_id is None:
            return []

        # Retrieve user data from the database
        user, profiles, stories = await self._query_user_data()
        current_knowledge_base = {
            "user_info": user,
            "profiles": profiles,
            "stories": stories,
        }

        files_paths = await convert_user_info_to_json_files(
            current_knowledge_base,
            self.config.output_artifacts.knowledge_base_root_dir,
        )
        # Ensure files_paths is a list, even if it's empty
        return files_paths or []

    async def _query_user_data(self) -> Tuple[User, List[Profile], List[Story]]:
        """
        Queries the database asynchronously for user information, profiles, and stories
        using SQLAlchemy.

        Returns:
            Tuple[User, List[Profile], List[Story]]: A tuple containing user information, profiles, and stories.
        """
        # Fetch user information
        user = await self.database_manager.fetch_user_by_id(self.user_id)

        # Fetch profiles and stories associated with the user
        profiles, stories = (
            await self.database_manager.fetch_profiles_and_stories_for_user(
                self.user_id
            )
        )

        return user, profiles, stories

    async def _process_story_generation_step(
        self, current_step: StoryState, **kwargs
    ) -> None:
        """
        Processes a given step and moves to the next step in the story generation process.

        Args:
            current_step (StoryState): The current step to be processed.
        """
        while current_step is not StoryState.FINAL_DOCUMENT_GENERATED:
            method_name = self.steps_execution_mapping.get(current_step)
            if method_name and hasattr(self, method_name):
                try:
                    await getattr(self, method_name)(**kwargs)

                    # Refresh here after the step update to ensure state consistency
                    await self.story_manager.refresh()

                    # Move to the next step
                    current_step = StoryState.next(current_step)
                    await self.update_story_state(current_step)
                except Exception as e:
                    logger.error(
                        f"Failed to execute step {method_name}: {traceback.format_exc()}"
                    )
                    break
            else:
                logger.error(f"No method found for step {current_step}")
                break

        if current_step is StoryState.FINAL_DOCUMENT_GENERATED:
            logger.info("Story generation process completed.")

    async def refresh_token_periodically(self):
        while True:
            try:
                # Refresh the token
                self.new_token = await refresh_access_token(self.new_token)
                logger.info("Access token refreshed.")

                # Sleep for the specified interval
                await asyncio.sleep(self.token_refresh_interval)
            except asyncio.CancelledError:
                # Handle the task cancellation gracefully
                logger.info("Token refresh task cancelled.")
                break
            except Exception as e:
                logger.error(f"Failed to refresh access token: {e}")
                break

    async def _reset(self) -> None:
        """
        Reset the variables for a fresh run.

        This method sets all internal state variables to their initial values.
        """
        try:
            # Resetting the StoryManager object to its initial state
            await self.story_manager.reset()

        except Exception as e:
            logger.error(
                f"An error occurred while resetting the story manager: {traceback.format_exc()}"
            )

    async def update_story_state(self, step: StoryState) -> None:
        """
        Update the story's state to the given step and commit changes through StoryManager.

        Args:
            step (StoryState): The new state to set for the story.

        Raises:
            ValueError: If the story manager or story is not loaded.
        """

        if not self.story_manager or not self.story_manager.story:
            error_msg = "No story manager or story loaded to update step."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Updating story state to {step.name}")
        await self.story_manager.update_story_step(step)
        logger.info(f"Story state updated to {step.name}")

    def _generate_starting_message(self, try_mode: bool = False) -> str:
        """
        Generates a starting message for the user based on the presence of a knowledge base.

        Returns:
            str: The starting message for the user.
        """
        if try_mode:
            return self.config.updates_request_prompts.starting_message_new_user
        else:
            return self.config.updates_request_prompts.starting_message_existing_user

    async def handle_assistant_requests(self, ai_message_for_system: dict) -> None:
        """
        Processes the Chat AI Assistant requests/commands.
        """
        logger.info(
            f"Processing AI message for Magic-Tales Orchestrator: {ai_message_for_system}"
        )

        if not isinstance(ai_message_for_system, dict):
            logger.info(
                f"AI message for Magic-Tales Orchestrator is not a dict: {ai_message_for_system}"
            )
            return

        command = ai_message_for_system.get("command", None)

        if not command:
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.no_command
            )
            return

        if command == Command.START_STORY_GENERATION:
            await self._handle_start_story_generation(ai_message_for_system)
            return

        if (
            command == Command.CONTINUE_STORY_GENERATION
            and self.wait_for_response_to_continue_where_we_left_off
        ):
            await self._handle_continue_where_we_left_off_response(
                ai_message_for_system
            )
            return

        if command == Command.UPDATE_PROFILE:
            if await self._update_profile_from_chat(ai_message_for_system):
                files_paths = await self._fetch_user_data_and_update_knowledge_base()
                await self.chat_assistant.update_assistant(files_paths=files_paths)
                await self._generate_and_send_update_message_for_user(
                    self.config.updates_request_prompts.profile_updated
                )
                await self.story_manager.refresh()
                # Construct a response to be sent by the AI Core Interface Layer to the Front End
                ai_response = WSOutput(
                    command=Command.PROFILE_UPDATED,
                    token=self.new_token,
                    working=False,  # Indicates that the response is ready
                 )
                await self.send_message_to_frontend(ai_response)
            return

        if command == Command.NEW_PROFILE:
            if await self._create_new_profile_from_chat(ai_message_for_system):
                files_paths = await self._fetch_user_data_and_update_knowledge_base()
                await self.chat_assistant.update_assistant(files_paths=files_paths)
                await self._generate_and_send_update_message_for_user(
                    self.config.updates_request_prompts.profile_created
                )
                await self.story_manager.refresh()
            return

    async def _check_for_all_info_to_start_story_generation(
        self, ai_message_for_system: dict
    ) -> bool:
        """
        Checks if the AI message contains all required information to start story generation.

        Args:
            ai_message_for_system (dict): The message_for_system dictionary from the AI response.

        Returns:
            bool: True if all required information is present, False otherwise.
        """

        required_keys = {
            "profile_id"
        }  # , "story_features", "story_synopsis", "story_title", "num_chapters"}
        missing_keys = required_keys - set(ai_message_for_system.keys())

        if missing_keys:
            logger.error(f"Missing required keys in message_for_system: {missing_keys}")

            # Craft a clear message to the assistant, specifying the missing keys
            update_message = self.config.updates_request_prompts.missing_keys_at_start_story_generation.format(
                missing_keys=", ".join(missing_keys)
            )
            await self._generate_and_send_update_message_for_user(update_message)

            return False

        # Additional Validation:
        profile_id = ai_message_for_system["profile_id"]

        try:
            profile_id = int(profile_id)  # Ensure profile_id is an integer
        except ValueError:
            logger.error("Invalid profile_id: Expected an integer")
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.invalid_profile_id_at_start_story_generation
            )
            return False

        # # Additional Validation: (NOTE: I'm missing the max num of chapters on the assistant instructions.)
        # num_chapters = ai_message_for_system["num_chapters"]

        # try:
        #     num_chapters = int(num_chapters)  # Ensure num_chapters is an integer and is between 1-10
        #     if not 1 <= num_chapters <= self.config.output_artifacts.max_num_chapters:
        #         raise ValueError
        # except ValueError:
        #     logger.error(f"Invalid num_chapters: Expected an integer that also must between 1-{self.config.output_artifacts.max_num_chapters}")
        #     update_message = self.config.updates_request_prompts.invalid_num_chapters_at_start_story_generation.format(max_num_chapters=", ".join(self.config.output_artifacts.max_num_chapters)
        #     )
        #     asyncio.create_task(self._generate_and_send_update_message_for_user(update_message)

        #     return False

        # Fetch and validate the profile
        current_profile = await self.database_manager.fetch_profile_by_id(profile_id)
        if not current_profile:
            logger.error(f"No profile found with ID: {profile_id}")
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.profile_id_does_not_exist
            )
            return False

        return True

    async def _handle_start_story_generation(self, ai_message_for_system: dict) -> None:
        """
        Handles the start_story_generation command from the AI assistant.
        """
        logger.info("start_story_generation request.")
        ok_to_proceed = await self._check_for_all_info_to_start_story_generation(
            ai_message_for_system
        )

        if ok_to_proceed:
            logger.info("Updating the database with new story elements from chat")
            ## story_info_dict = ai_message_for_system
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.chat_info_extraction
            )
            story_info_dict = await self._extract_chat_key_elements(
                await self.chat_assistant._retrieve_messages()
            )
            story_info_dict["profile_id"] = ai_message_for_system["profile_id"]
            await self.create_story_foundation_from_chat_elements(story_info_dict)
            logger.info("Database updated with new story elements from chat")
            await self._process_story_generation_step(StoryState.STORY_GENERATION)

        # If we get here, we must have finished the story!
        return

    async def _handle_continue_where_we_left_off_response(
        self, ai_message_for_system: dict
    ) -> None:
        """
        Handles whether the user want to continue an unfinished story where we left it off, or not.
        """
        continue_where_we_left_off = ai_message_for_system.get(
            "continue_where_we_left_off", None
        )
        logger.info(
            f"'Continue where we left off' RESPOSE received:{continue_where_we_left_off}"
        )

        # Ensure that we got a proper response
        if (
            continue_where_we_left_off is None
            or continue_where_we_left_off.lower() not in ("true", "false")
        ):
            logging.error(
                "'Continue where we left off' command by the assistant had no response. Asking assistant to solve it"
            )
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.continue_where_we_left_off_response_missing_or_incorrect
            )
            return

        self.wait_for_response_to_continue_where_we_left_off = False
        if continue_where_we_left_off == "true":
            await self._resume_story()
        else:
            asyncio.create_task(self._handle_new_tale())

    async def _update_profile_from_chat(self, ai_message_for_system: dict) -> bool:
        """
        Updates a profile based on the AI system message.

        Args:
            ai_message_for_system (dict): The message from the AI system with update details.
        """
        logger.info("Atempting to update profile...")
        profile_id = ai_message_for_system.get("profile_id", None)

        if not profile_id:
            logging.error(
                "Profile ID is required for profile update, but not provided by the assistant. Asking assistant to solve"
            )
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.no_profile_id_at_update_profile
            )
            return False

        profile = await self.database_manager.session.get(Profile, profile_id)
        if not profile:  # Profile record does not Exist, most likely wrong ID
            logging.error(
                "Profile record with this id does not exist on the DB. Asking assistant to solve"
            )
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.no_profile_record_at_update_profile
            )
            return False

        logger.info(f"Profile extracted:{profile_id}")

        name = ai_message_for_system.get("name", None)
        age = ai_message_for_system.get("age", None)
        details = ai_message_for_system.get("details", None)

        # Ensure there there is at least something to update
        if not name or not age or not details:
            logging.error(
                "Name, age and details are required for profile update, but one or more were NOT provided by the assistant. Asking assistant to solve"
            )
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.no_info_at_update_profile
            )
            return

        logger.info(f"profile UPDATES:\nName:{name}\nAge:{age}\nDetails:{details}")

        # Fetch existing profile details
        # current_profile = await self.database_manager.fetch_profile_by_id(profile_id)
        # existing_details = current_profile.details if current_profile else ""
        # logger.info(f"current profile details:\n{existing_details}")

        # # Merge updates using the Information Extractor Assistant
        # merged_details = await self.merge_profile_updates(existing_details, updates)
        # logger.info(f"merged profile details:\n{merged_details}")

        try:
            await self.database_manager.update_profile_record_by_id(
                profile_id, {"name": name, "age": age, "details": details}
            )
            await self.story_manager.refresh()
            return True
        except Exception as e:
            logger.error(f"Failed to Update the profile: {traceback.format_exc()}")
            return False

    async def _create_new_profile_from_chat(self, ai_message_for_system: dict) -> bool:
        """
        Creates a new profile based on the AI system message.

        Args:
            ai_message_for_system (dict): The message from the AI system with update details.
        """
        logger.info("Atempting to create a new profile...")
        name = ai_message_for_system.get("name", None)
        age = ai_message_for_system.get("age", None)
        details = ai_message_for_system.get("details", None)

        # Ensure there there is at least something to update
        if not name or not age or not details:
            logging.error(
                "Name, age and details are required to create a new profile, but none were provided by the assistant. Asking assistant to solve"
            )
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.no_info_at_create_profile
            )
            return False

        logger.info(f"profile UPDATES:\nName:{name}\nAge:{age}\nDetails:{details}")

        try:
            await self.database_manager.create_profile_from_chat_details(
                {"name": name, "age": age, "details": details, "user_id": self.user_id}
            )
            await self.story_manager.refresh()
            return True
        except Exception as e:
            logger.error(f"Failed to Create the profile: {traceback.format_exc()}")
            return False

    async def merge_profile_updates(self, existing_details: str, updates: str):
        """
        Merges updates into the existing profile details using the Information Extractor.

        Args:
            existing_details (str): The current details of the profile in JSON format.
            updates (str): The updates to the profile in JSON format.

        Returns:
            str: The merged profile details in JSON format.
        """
        # Convert existing details and updates into a format suitable for the extractor.
        chat_string = (
            f"Existing details: {existing_details}\nUpdates: {updates}\n"
            f"Merge these updates into the existing details."
        )

        # Use the Information Extractor to merge the details.
        merged_details, _ = await self.helper_assistant.request_ai_response(chat_string)
        return merged_details

    async def _convert_chat_to_string(self, messages: List[Any]) -> str:
        """
        Converts a list of chat messages into a single concatenated string, focusing on text content.

        Args:
            messages (List[ThreadMessage]): The list of chat messages, each containing a list of `Content` objects.

        Returns:
            str: A concatenated string of all textual chat messages.
        """
        chat_lines = []
        for msg in messages:
            for content in msg.content:
                # Check if the content is of type TextContentBlock and extract text
                if isinstance(content, TextContentBlock):
                    chat_lines.append(f"{msg.role}: {content.text}")
                # Optionally handle ImageFileContentBlock types differently, e.g., by noting an image was present
                elif isinstance(content, ImageFileContentBlock):
                    chat_lines.append(f"{msg.role}: [Image file content]")
                # Optionally handle ImageURLContentBlock types differently, e.g., by noting an image was present
                elif isinstance(content, ImageURLContentBlock):
                    chat_lines.append(f"{msg.role}: [Image URL content]")
        return "\n".join(chat_lines)

    async def _extract_chat_key_elements(self, chat_string: str) -> Dict:
        """
        Asynchronously extracts key story elements from the chat string.

        Args:
            chat_string (str): The concatenated chat messages as a single string.

        Returns:
            Dict: A dictionary containing extracted elements.
        """
        story_features, _ = await self.helper_assistant.request_ai_response(
            f"Identify and extract a list of all story features from this conversation, including, but not limited to genre, target audience, language, story length (time, chapters, word count), themes, style, setting, tone, plot points, relevant literary elements etc. Here's the conversation:\n{chat_string}"
        )

        story_synopsis, _ = await self.helper_assistant.request_ai_response(
            f"Extract the latest agreed upon with the user story synopsis in full detail from this conversation. Make sure you capture all the latest details about characters, plot, theme, story flow, language, etc. Do not miss anything that has been agreed upon, as all these details are crucial for the generation of the perfect story. Also review and add to the synopsiis all extra information about the characters, scenes, anything that it was agreed upon on the conversation that could bring more details about what the user wanted. Here's the conversation:\n{chat_string}"
        )

        story_title, _ = await self.helper_assistant.request_ai_response(
            f"Extract only the story title from this synopsis without any highligthing character such * or #:\n{story_synopsis}.\n\nThe title is:"
        )

        num_chapters, _ = await self.helper_assistant.request_ai_response(
            f"Extract only the number of chapters, as a single integer, from this synopsis and story features gathred from a conversation. If unknown, deduce what would be an ideal number of chapters between 1 and {self.config.output_artifacts.max_num_chapters} for this story to unfold fully:\nSynopsis:{story_synopsis}\nStory Features:{story_features}.\n\nThe number of chapters is (only an integer, no str allowed):"
        )

        extracted_info = {
            "story_features": story_features,
            "story_synopsis": story_synopsis,
            "story_title": story_title,
            "num_chapters": num_chapters,
        }
        logger.info(
            f"We have extracted the following information from the chat:\n{extracted_info}"
        )
        return extracted_info

    async def create_story_foundation_from_chat_elements(self, updates: dict):
        """
        Creates the foundational data for a story based on the key elements obtained from chat.

        Args:
            updates (dict): A dictionary containing the elements ('personality_profile',
                            'story_features', 'story_synopsis', 'story_title') required to create a story.

        Raises:
            ValueError: If required profile details are missing or incomplete.
            Exception: If unable to create a story due to database or validation errors.
        """
        logger.info(f"The information we got to create the new story is:\n{updates}")

        # Extract story elements from updates
        profile_id = updates.get("profile_id")
        title = updates.get("story_title")
        features = updates.get("story_features")
        synopsis = updates.get("story_synopsis")
        num_chapters = updates.get("num_chapters")

        # Define paths for story and image storage based on user-specific directories
        stories_root_dir = os.path.join(
            self.config.output_artifacts.stories_root_dir, f"user_{self.user_id}"
        )

        story_folder, images_subfolder = create_new_story_directory(
            stories_root_dir, subfolders=self.subfolders
        )

        # Create the story using the StoryManager class
        try:
            await self.story_manager.create_story(
                profile_id=profile_id,
                # ws_session_uid=self.session.identity_key,  # Session identity key needs to be accurately sourced
                ws_session_uid=self.websocket.uid,           # Websocket session UID associated with the story. (self.session.identity_key returns a reference to the method, not a valid attribute)
                title=title,
                features=features,
                synopsis=synopsis,
                story_folder=story_folder,
                images_subfolder=images_subfolder,
            )
            self.story_manager.in_mem_story_data.num_chapters = max(int(num_chapters), 1)

            logger.info(
                f"Story foundation created successfully with ID: {self.story_manager.story.id}"
            )
        except Exception as e:
            logger.error(f"Failed to create story foundation: {e}")
            raise Exception(
                "Failed to create the story foundation due to an unexpected error."
            ) from traceback.format_exc()

    async def _generate_chapter_title(self, chapter: str) -> str:
        """Generate a title for specific Chapter"""
        logger.info(f"Generating a title for the Chapter")
        message = f"Given the following piece of information:\n\n1) Chapter: {chapter}.\n\nThe best possible title for this chapter is (just respond with the title, nothing else):"
        chapter_title, _ = await self.helper_assistant.request_ai_response(message)
        logger.info(f"Chapter Title: {chapter_title}")
        return chapter_title

    async def _generate_story(self) -> None:
        """
        Generate a story based on the provided inputs.

        This method ensures that all necessary components for the story are available,
        and it handles failures to ensure that the story generation process either
        completes successfully or provides clear diagnostics for any issues encountered.


        Generates:
            List[Dict[str, str]]: A list of dictionaries, each containing the chapter title and content.

        Resturns:
            None
        """
        # Attempt to refresh story and profile data
        try:
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.generating_story
            )

            await self.story_manager.refresh(raise_error=True)
            story_blueprint = await self.story_manager.get_story_blueprint()

        except Exception as e:
            logger.error(f"Failed to initialize story generation: {e}")
            if self.latest_story and self.latest_story.id:
                try:
                    await self.story_manager.load_story(self.latest_story.id)
                    story_blueprint = await self.story_manager.get_story_blueprint()
                except Exception as load_error:
                    logger.error(f"Failed to load the latest story: {load_error}")
                    raise RuntimeError(
                        "Unable to proceed with story generation due to initialization failure."
                    )
            else:
                raise RuntimeError(
                    "No latest story available to fallback to and cannot generate new story."
                )

        # Initialize a list to hold the chapters
        chapters = []

        # Step 1: Chapter Count
        num_chapters = int(self.story_manager.in_mem_story_data.num_chapters)

        logger.info(f"Number of chapters: {num_chapters}")

        # Step 2: Initialize chapter content
        previous_chapter_content = ""

        # Step 3: Chapter Generation
        chapters_folder = os.path.join(
            self.story_manager.story.story_folder, self.subfolders["chapters"]
        )

        for chapter_number in range(1, num_chapters + 1):
            message = f"In a few words update the user that we are now, generating chapter: {chapter_number} of {num_chapters}"
            logger.info(message)
            await self._generate_and_send_update_message_for_user(message)

            chapter_title, chapter_content = await self._generate_chapter(
                chapters_folder=chapters_folder,
                story_blueprint=story_blueprint,
                chapter_number=chapter_number,
                total_number_chapters=num_chapters,
                previous_chapter_content=previous_chapter_content,
            )
            # Add the generated chapter to the chapters list
            chapters.append({"title": chapter_title, "content": chapter_content})
            previous_chapter_content = chapter_content

            await self.create_and_send_progress_update(
                story_state=str(StoryState.STORY_GENERATION),
                status=ResponseStatus.STARTED,
                progress_percent=0.3 * (chapter_number / num_chapters) * 100,
                # message="...",
            )

        # Update in-memory story data
        self.story_manager.in_mem_story_data.chapters = chapters
        return

    async def _generate_chapter(
        self,
        chapters_folder: str,
        story_blueprint: Dict[str, str],
        chapter_number: int,
        total_number_chapters: int,
        previous_chapter_content: str,
    ) -> tuple:
        """
        Generate a single chapter of the story.

        Args:
            chapter_number (int): The index of the chapter to generate.
            total_number_chapters (int): The total number of chapters in the story.
            previous_chapter_content (str): The latest chapter of the story.

        Returns:
            tuple: The generated chapter title and content.
        """
        try:
            self.chapter_generation_mechanism = ChapterGenerationMechanism(
                config=self.config,
                story_blueprint=story_blueprint,
                previous_chapter_content=previous_chapter_content,
            )
            best_chapter = self.chapter_generation_mechanism.create_chapter(
                chapters_folder=chapters_folder,
                chapter_number=chapter_number,
                total_number_chapters=total_number_chapters,
            )
            chapter_content = best_chapter.get(
                "chapter_generator_response_dict", {}
            ).get("content", "")
            if chapter_content == "":
                raise Exception("Failed to generate a chapter. Please try again.")

            if total_number_chapters > 1:
                # Extract the chapter title
                chapter_title = await self._generate_chapter_title(chapter_content)
            else:
                chapter_title = ""

            logger.info(f"Chapter {chapter_number}: {chapter_title}")
            logger.info(f"{chapter_content}")

            return chapter_title, chapter_content
        except Exception as e:
            logger("There's been and error generating the Chapter")

    async def _generate_image_prompts_for_all_chapters(
        self, chapters: List[Dict[str, str]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate image prompts for all chapters in the story.
        """
        logger.info("Generating image prompts for all chapters.")
        all_image_prompts = {}
        num_chapters = len(chapters)
        for i, chapter_dict in enumerate(chapters):
            try:
                chapter_number = i + 1
                chapter_title = chapter_dict.get("title", f"Chapter {chapter_number}")
                chapter_content = chapter_dict.get("content", "")
                logger.info(
                    f"Generating image prompts for Chapter {chapter_number}: {chapter_title}"
                )

                image_prompt_data = self.image_prompt_generation_mechanism._generate_image_prompts_per_chapter(
                    chapter_number, chapter_content
                )
                if image_prompt_data.get("image_prompt_generator_success"):
                    all_image_prompts[i] = {
                        "title": chapter_title,
                        "image_prompt_data": image_prompt_data,
                    }
                else:
                    logger.warning(
                        f"Failed to generate image prompts for {chapter_title}. Skipping."
                    )
                await self.create_and_send_progress_update(
                    story_state=str(StoryState.IMAGE_PROMPT_GENERATION),
                    status=ResponseStatus.STARTED,
                    progress_percent=30.0 + 0.3 * (chapter_number / num_chapters) * 100,
                    # message="...",
                )
            except Exception as e:
                logger.error(
                    f"Exception while generating image prompts for {chapter_title}: {e}",
                    exc_info=True,
                )

        return all_image_prompts

    async def _generate_image_prompts(self) -> None:
        """Generate image prompts for all chapters in the story."""

        message = f"In a few words, tell the user we have entered a new phase in the story generation process. We are now Generating vivid and amazing image descriptions for all chapters."
        logger.info(message)
        await self._generate_and_send_update_message_for_user(message)

        self.image_prompt_generation_mechanism = ImagePromptGenerationMechanism(
            config=self.config
        )
        all_chapter_prompts_and_new_chapters_annotated = (
            await self._generate_image_prompts_for_all_chapters(
                chapters=self.story_manager.in_mem_story_data.chapters
            )
        )
        await self._extract_post_processed_chapters(
            all_chapter_prompts_and_new_chapters_annotated
        )
        logger.info(
            f"Image Prompts + processed chapters created"
        )  # {self.story_manager.in_mem_story_data.image_prompts}")

    async def _extract_post_processed_chapters(
        self, all_image_prompts: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Save the post-processed chapters, which include image annotations, into the InMemStoryData structure.

        Args:
            all_image_prompts: A dictionary containing image prompt data for each chapter.

        Returns:
            None
        """
        post_processed_chapters = []
        image_prompt_messages = []
        image_prompts = []

        for chapter_number, chapter_data in all_image_prompts.items():
            chapter_title = chapter_data.get("title", f"Chapter {chapter_number + 1}")
            chapter_content = chapter_data["image_prompt_data"][
                "image_prompt_response_content_dict"
            ]["annotated_chapter"]
            image_prompt = chapter_data["image_prompt_data"][
                "image_prompt_response_content_dict"
            ]["image_prompts"]
            prompt_messages = chapter_data["image_prompt_data"][
                "image_prompt_generator_prompt_messages"
            ]

            post_processed_chapters.append(
                {"title": chapter_title, "content": chapter_content}
            )

            image_prompts.append(image_prompt)
            image_prompt_messages.append(
                {
                    "system_message": prompt_messages[0].content,
                    "human_message": prompt_messages[1].content,
                }
            )

        # Save the post-processed chapters to the InMemStoryData structure.
        self.story_manager.in_mem_story_data.post_processed_chapters = (
            post_processed_chapters
        )
        self.story_manager.in_mem_story_data.image_prompt_generator_prompt_messages = (
            image_prompt_messages
        )
        self.story_manager.in_mem_story_data.image_prompts = image_prompts

    async def _generate_images(self) -> List[str]:
        """
        Generate images based on provided image prompts for all chapters and save them to a specified directory.

        Returns:
            A list of filenames for the generated images.

        Raises:
            Exception: If any error occurs during the image generation process.
        """

        logger.info("Initiating image generation process.")
        message = f"Update the user in just a few words that we have entered the third of four phases in the story generation process. We are now going to start creating all the illustrations."
        await self._generate_and_send_update_message_for_user(message)

        # await self.story_manager.refresh()

        all_image_prompts = self.story_manager.in_mem_story_data.image_prompts

        # Calculate total images to be generated for Updates. Ensuring at least 1
        total_number_images = max(
            sum(len(chapter_prompts) for chapter_prompts in all_image_prompts), 1
        )
        logger.warn(f"Number of images to generate:{total_number_images}")

        story_directory = self.story_manager.story.story_folder

        # Create a subdirectory for images within the specified story directory
        image_directory = os.path.join(story_directory, self.subfolders["images"])
        os.makedirs(image_directory, exist_ok=True)

        # Initialize image generator with the provided configuration
        # image_generator = instantiate(self.config.image_generator)
        image_generator = DALLE3ImageGenerator(self.config.image_generator)

        # Initialize list to store filenames of generated images
        image_filenames = []

        # Flatten list of lists and keep track of chapter numbers
        flat_image_prompts = [
            (chapter_number, image_prompt, image_prompt_index)
            for chapter_number, chapter_prompts in enumerate(all_image_prompts, 1)
            for image_prompt_index, image_prompt in enumerate(chapter_prompts)
        ]

        image_generation_count = 0
        for chapter_number, image_prompt, image_prompt_index in flat_image_prompts:
            filename = f"Chapter_{chapter_number}_Image_{image_prompt_index}.png"
            message = f"Update the user in just a few words that we are Generating Chapter {chapter_number}, illustration {image_prompt_index+1}"
            logger.info(message)
            await self._generate_and_send_update_message_for_user(message)

            try:
                image_generation_count += 1
                generated_image_path = image_generator.generate_images(
                    [image_prompt], image_directory
                )

                if generated_image_path:
                    os.rename(
                        generated_image_path[0], os.path.join(image_directory, filename)
                    )
                    image_filenames.append(filename)
                else:
                    message = f"Failed to generate image for Chapter {chapter_number}, Image {image_prompt_index}. Skipping."
                    logger.error(message)

                await self.create_and_send_progress_update(
                    story_state=str(StoryState.IMAGE_GENERATION),
                    status=ResponseStatus.STARTED,
                    progress_percent=60.0
                    + 0.3 * (image_generation_count / total_number_images) * 100,
                    # message="...",
                )

            except Exception as e:
                message = f"An error occurred while processing Chapter {chapter_number}, Image {image_prompt_index}."
                logger.error(f"{message}. Details:\n{traceback.format_exc()}")
                continue

        logger.info(f"Successfully generated image files: {image_filenames}")
        return image_filenames

    async def _filter_chat_info(
        self, chat_assistant_info: List[Tuple[str, int]], info_to_extract: int
    ) -> str:
        """
        Filters and creates a str of the chat information based on the specified info type.

        Args:
            chat_assistant_info (List[Tuple[str, int]]): A list of messages generated by the "openai_assistant".
                Each tuple consists of a message (str) and its associated data bucket type (int).
            info_to_extract (int): The information type to extract and summarize.
                1: Personality Profile, 2: Story Features, 3: Story Synopsis

        Returns:
            str: A string of the filtered messages.
        """

        # Filter messages by the specified info type
        filtered_messages = [
            message
            for message, data_type in chat_assistant_info
            if data_type == info_to_extract
        ]

        # Create a str of the filtered messages
        return "\n".join(filtered_messages)

    async def _generate_final_document(self) -> NoReturn:
        """
        Asynchronously creates the final story document with associated images and saves it to the specified directory.
        """
        message = f"Update the user in a few words that we just entered the last phase of the story generation process. We are now starting the final document generation, where we put everything together: All Chapters and all images, and we create a downloadable file."
        logger.info(message)
        await self._generate_and_send_update_message_for_user(message)

        try:
            story_document = StoryDocument(
                title=self.story_manager.story.title,
                chapters=self.story_manager.in_mem_story_data.post_processed_chapters,
                image_filenames=self.story_manager.in_mem_story_data.image_filenames,
                story_folder=self.story_manager.story.story_folder,
                images_subfolder=self.story_manager.story.images_subfolder,
            )

            story_document.create_document()
            await self.create_and_send_progress_update(
                story_state=str(StoryState.DOCUMENT_GENERATION),
                status=ResponseStatus.STARTED,
                progress_percent=93.3,
                # message="...",
            )
            doc_filepath = story_document.save_document("story.docx")
            await self.create_and_send_progress_update(
                story_state=str(StoryState.DOCUMENT_GENERATION),
                status=ResponseStatus.STARTED,
                progress_percent=96.6,
                # message="...",
            )
            final_pdf_file_path = story_document.convert_docx_to_pdf(doc_filepath)
            await self.create_and_send_progress_update(
                story_state=str(StoryState.DOCUMENT_GENERATION),
                status=ResponseStatus.STARTED,
                progress_percent=100.0,
                # message="...",
            )
            logger.info(f"Story document saved in: {final_pdf_file_path}")
            message = "In a few words, let the user know that we have finished and that the entire team at Magic-Tales.ai hope they enjoy this story!"
            await self._generate_and_send_update_message_for_user(message)

            http_base_url = "http://localhost:8000"
            await self.create_and_send_progress_update(
                command=Command.PROCESS_COMPLETED,
                story_state=str(StoryState.DOCUMENT_GENERATION),
                files=[http_base_url + final_pdf_file_path],
                data={"story_id": self.story_manager.story.id},
                progress_percent=100.0,
            )

            files_paths = await self._fetch_user_data_and_update_knowledge_base()
            await self.chat_assistant.update_assistant(files_paths=files_paths)

        except Exception as e:
            message = "In a few words, let the user know that we have failed to generate final document and that the entire team at Magic-Tales.ai apologizes and it's working very hard to find out what exactly happened and fix it soon."
            logger.error(message, exc_info=True)
            raise (f"{message}: {e}\n{traceback.format_exc()}")
        finally:
            await self.send_working_command_to_frontend(False)
