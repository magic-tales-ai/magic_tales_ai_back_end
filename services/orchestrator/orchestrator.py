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
from sqlalchemy.exc import SQLAlchemyError, NoResultFound
from openai.types.beta.threads import MessageContentText
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

from models.story_state import StoryState
from models.story import Story
from .story_manager import StoryManager
from .database_manger import DataManager
from models.command import Command
from models.response import ResponseStatus
from models.message import Message, MessageSchema, OriginEnum, TypeEnum
from models.profile import Profile
from models.user import User
from models.ws_input import WSInput
from models.ws_output import WSOutput


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
        StoryState.USER_FACING_CHAT: "_user_facing_chat",
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

        self.message_sender = message_sender
        self.session = session
        self.websocket = websocket
        self.token_data = None

        self.user_id = None
        self.user_dict = {}
        self.profile_id = None
        self.latest_story = None

        self.new_token = None

        self.token_refresh_interval = 120  # 2 minutes in seconds
        self.token_refresh_task = None

        self.chat_completed = False

        self.story_manager = StoryManager(session=session)

        self.database_manger = DataManager(session=session)

        logger.info("MagicTales initialized.")

    def _validate_openai_api_key(self) -> None:
        """Validates the presence of the OpenAI API key."""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    async def process_request(self, request: WSInput, token_data: dict) -> None:
        """
        Process incoming commands/requests from the AI Core Interface Layer ().

        Args:
            request (WSInput): Full command structure. Please look at the ICD docs for more details.
            token_data (dict): Token data from the AI Core Interface Layer.

        Returns:
            dict: Response data to be sent back to the AI Core Interface Layer.
        """
        # logger.info(f"token:{request.token}")  # DEBUG

        self.token_data = token_data
        if not self.token_refresh_task:
            self.new_token = await refresh_access_token(request.token)
            self.token_refresh_task = asyncio.create_task(
                self.refresh_token_periodically()
            )

        # No matter what We will always have a user!!
        coming_user_id = token_data.get("user_id", None)

        # Send ACK if command received requires it
        if (
            request.command == Command.NEW_TALE
            or request.command == Command.SPIN_OFF
            or request.command == Command.UPDATE_PROFILE
        ):
            await self.send_message_to_frontend(
                WSOutput(command=request.command, token=self.new_token, ack=True)
            )

        await self.send_working_command_to_frontend(True)

        # There has been a change in the user_id
        if self.user_id != coming_user_id:
            self.user_id = coming_user_id
            self.user_dict = (await self.database_manger.fetch_user_by_id(self.user_id)).to_dict()

            user_message = self._generate_starting_message(try_mode=request.try_mode)
            if self.user_dict:
                logger.info(f"User {self.user_id} has been set.")

                assistant_id = self.user_dict.get("assistant_id", None)
                helper_id = self.user_dict.get("helper_id", None)

                await self.chat_assistant.start_assistant(assistant_id)
                await self._generate_and_send_update_message_for_user(user_message)
                if assistant_id != self.chat_assistant.openai_assistant.id:
                    logger.warn(
                        f"User {self.user_id} had CHAT assistant {assistant_id}, which was not found. It's going to be updated. This should only happen with brand new users!"
                    )
                    await self.database_manger.update_user_record_by_id(
                        self.user_id,
                        {"assistant_id": self.chat_assistant.openai_assistant.id},
                    )

                await self.helper_assistant.start_assistant(helper_id)
                if helper_id != self.helper_assistant.openai_assistant.id:
                    logger.warn(
                        f"User {self.user_id} had HELPER assistant {helper_id}, which was not found. It's going to be updated. This should only happen with brand new users!"
                    )
                    await self.database_manger.update_user_record_by_id(
                        self.user_id,
                        {"helper_id": self.helper_assistant.openai_assistant.id},
                    )

                await self._fetch_user_data_and_update_knowledge_base()
                await self._generate_and_send_update_message_for_user(
                    self.config.updates_request_prompts.after_feching_user_data
                )
            else:
                logger.warn(f"User {self.user_id} has not been found.")
                raise (f"User {self.user_id} has not been found.")

        await self.send_working_command_to_frontend(False)

        # CONVERSATION FOR ALL COMMANDS
        conversation = Message(
            user_id=self.user_id,
            session_id=self.websocket.uid,
            command=request.command,
            origin=OriginEnum.user,
            type=(
                TypeEnum.chat
                if request.command == Command.USER_MESSAGE
                else TypeEnum.command
            ),
            details=request.model_dump(),
        )
        await self.database_manger.add_message_to_session_record(conversation)

        ## HANDLEING EACH COMMAND POSSIBILITY
        if request.command == Command.NEW_TALE:
            asyncio.create_task(
                self.check_last_story_and_allow_to_continue(self.user_id)
            )
            if self.latest_story is None:
                asyncio.create_task(self._handle_new_tale())

        elif request.command == Command.SPIN_OFF:
            if not request.story_id:
                raise Exception("story_id is required for spin-off")
            asyncio.create_task(self._handle_spin_off_tale(request))

        elif request.command == Command.UPDATE_PROFILE:
            if not request.profile_id:
                raise Exception("profile_id is required for update profile")

            await self.send_working_command_to_frontend(True)
            asyncio.create_task(self._handle_user_request_update_profile(request))

        # CONVERSATION RECOVERY
        # this command is necesary to recover a current and not finished conversation
        # the recover is based on CHAT TABLE records
        #
        elif request.command == Command.CONVERSATION_RECOVERY:
            if not self.websocket.uid:
                raise Exception(
                    "uid is required for conversation recovery"
                )  # SESSION/CONVERSATION ID

            # read current conversation (CHAT ID)
            conversations = await self.database_manger.get_messages_by_session_id(self.websocket.uid)
            conversation_dicts = [
                MessageSchema().dump(conversation) for conversation in conversations
            ]
            # send full conversation to CLIENTE/USER (to rebuild it)
            await self.send_message_to_frontend(
                WSOutput(
                    command=request.command,
                    token=self.new_token,
                    data={"conversations": conversation_dicts},
                )
            )

        # ASSOCIATE USER WITH CONVERSATIONS
        elif request.command == Command.LINK_USER_WITH_CONVERSATIONS:
            if not self.user_id:
                raise Exception("user_id is required for link user with conversations")

            if not request.session_ids:
                raise Exception(
                    "session_ids is required for link user with conversations"
                )  # SESSION/CONVERSATION ID

            await self.database_manger.link_user_with_conversations_by_session_ids(request.session_ids)
            await self.send_message_to_frontend(
                WSOutput(command=request.command, token=self.new_token, ack=True)
            )

        elif request.command == Command.USER_MESSAGE:
            ai_response = await self._handle_user_message(request)
            await self.send_message_to_frontend(ai_response)

        elif request.command == Command.USER_LOGGED_IN:
            await self.check_last_story_and_allow_to_continue(self.user_id)

        logger.info(f"Processed command: {request.command}")

    async def fetch_latest_story_for_user(self, user_id):
        """
        Fetches the latest story that the user interacted with by leveraging the DataManager to reduce direct database calls.

        Args:
            user_id (int): The user ID whose latest story is being fetched.

        Returns:
            Story or None: The most recently updated story if available, otherwise None.

        Raises:
            Exception: If there is an error during the database operation or processing.
        """
        try:
            # Fetch all stories for the user using the DataManager
            stories = await self.database_manger.fetch_stories_for_user(user_id=user_id)

            # Determine the latest story based on the 'last_updated' field
            latest_story = max(stories, key=lambda story: story.last_updated, default=None)

            if latest_story:
                logger.info(f"Latest story for user {user_id} is story ID {latest_story.id} from profile ID {latest_story.profile_id}")
            else:
                logger.info(f"No stories found for user {user_id}.")

            return latest_story

        except Exception as e:
            logger.error(f"Error fetching the last story for user {user_id}: {str(e)}")
            return None

    async def check_last_story_and_allow_to_continue(self, user_id):
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
            else:
                logger.info(f"Last story completed step was: {self.latest_story.last_successful_step.name}"
                )
                # Process to resume the story
                await self._generate_and_send_update_message_for_user(
                    self.config.updates_request_prompts.incomplete_story_found
                )
        else:
            # Log that no story is available; await user action to start new.
            logger.info(
                "No story available for user; awaiting initiation of new story."
            )

    async def _resume_story(self):
        """
        Resumes the story generation process from the last saved state.
        """
        logger.info("Resuming story generation from the last saved state.")
        # Assume latest_story_dir contains necessary data to resume story.
        # You might need to adjust how you load story data.
        await self.story_manager.load_story_manager(self.latest_story.id)
        self.latest_story = None

        if self.story_manager.story:
            last_step = await self.story_manager.get_last_step()
            logger.info(f"Resuming story generation from step: {str(last_step)}")
            await self._process_story_creation_step(last_step)
        else:
            logger.info("No saved state found, awaiting user action.")
            # let's make sure everything is clean if we really didn't recover any story.. just in case
            self.story_manager.reset()

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
        message = Message(
            user_id=self.user_id,
            session_id=self.websocket.uid,
            command=output_model.command,
            origin=OriginEnum.ai,
            type=TypeEnum.chat,
            details=details_serialized,
        )

        await self.database_manger.add_message_to_session_record(message)
    

    async def create_progress_update(self, **kwargs) -> WSOutput:
        """
        Creates a progress update message in JSON format.

        This method dynamically constructs a progress update message based on the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments representing various components of the progress update message.

        Returns:
            WSOutput: The constructed progress update message.

        Example usage:
            create_progress_update(story_state="STORY_GENERATION", status=ResponseStatus.STARTED, progress_percent=50, message="Story generation in progress")
        """

        progress_update = WSOutput(
            command=Command.STATUS_UPDATE,
            token=self.new_token,
        )

        # Iterate through the keyword arguments and set them on the progress_update instance
        for key, value in kwargs.items():
            if hasattr(progress_update, key):
                setattr(progress_update, key, value)
            else:
                raise KeyError(f"Invalid progress update parameter: '{key}'")

        return progress_update

    async def _handle_user_message(self, request: WSInput) -> WSOutput:
        """
        Handle a user message by sending it to the chat assistant and processing the response.

        Args:
            user_message (str): The user's message to be processed.
            websocket (str): Unique identifier of the client.

        Returns:
            WSOutput: response to be sent back to the AI Core Interface Layer.
        """
        user_message = request.message

        # Generate AI response for the user message
        await self.send_working_command_to_frontend(True)
        ai_message_for_human, ai_message_for_system = (
            await self.chat_assistant.request_ai_response(message=user_message)
        )

        logger.info(f"AI response for human/user: {ai_message_for_human}")

        # Construct a response to be sent by the AI Core Interface Layer to the Front End
        response = WSOutput(
            command=Command.MESSAGE_FOR_HUMAN,
            message=ai_message_for_human,  # AI-generated response to be displayed to the user
            token=self.new_token,
            working=False,  # Indicates that the response is ready
        )

        if ai_message_for_system and isinstance(ai_message_for_system, WSInput):
            logger.info(
                f"AI Sent a COMMAND for the Orchestrator: {ai_message_for_system}"
            )
            await self.handle_assistant_requests(ai_message_for_system)

        await self.send_working_command_to_frontend(False)

        return response

    async def _handle_user_request_update_profile(self, request: WSInput) -> None:
        """
        Handle the 'user_request_update_profile' command by retrieving user information from the database, creating a chat where the user might night give us new information about this profile.

        Returns:
            None.
        """
        logger.info("Updating profile.")

        # self._reset()
        # await self._process_story_creation_step(StoryState.USER_FACING_CHAT)

    async def _handle_spin_off_tale(self, request: WSInput) -> None:
        """
        Handle the 'spin_off' command by retrieving user information from the database.

        Returns:
            None.
        """
        logger.info("Starting a spin-off story generation.")

        self._reset()
        await self._process_story_creation_step(StoryState.USER_FACING_CHAT)

    async def _handle_new_tale(self) -> None:
        """
        Handle the 'new-tale' command by retrieving user information from the database.

        Returns:
            None.
        """
        logger.info("Starting story generation from scratch.")
        self._reset()
        await self._process_story_creation_step(StoryState.USER_FACING_CHAT)

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
        ai_response = await self._handle_user_message(request_message)

        await self.send_message_to_frontend(ai_response)

    async def _fetch_user_data_and_update_knowledge_base(self) -> None:
        """
        Retrieves user data from the database and creates a knowledge base file.

        Returns:
            List[str]: List of file paths to the created knowledge base files.
        """
        if self.user_id is None:
            return

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
        files_paths = files_paths or []
        await self.chat_assistant.update_assistant(files_paths=files_paths)

    async def _query_user_data(self) -> Tuple[User, List[Profile], List[Story]]:
        """
        Queries the database asynchronously for user information, profiles, and stories
        using SQLAlchemy.

        Returns:
            Tuple[User, List[Profile], List[Story]]: A tuple containing user information, profiles, and stories.
        """
        # Fetch user information
        user = await self.database_manger.fetch_user_by_id(self.user_id)

        # Fetch profiles associated with the user
        profile_result = await self.session.execute(
            select(Profile).where(Profile.user_id == self.user_id)
        )
        profiles = profile_result.scalars().all()

        # Fetch stories associated with the user
        stories = await self.database_manger.fetch_stories_for_user(self.user_id)

        return user, profiles, stories

    def _should_resume(self) -> bool:
        """
        Determines if the story generation process should resume from the last saved state.

        Returns:
            bool: True if the story should be resumed, False otherwise.
        """
        self.latest_story_dir = get_latest_story_directory(
            self.config.output_artifacts.stories_root_dir
        )
        return (
            self.config.output_artifacts.continue_where_we_left_of
            and self.latest_story_dir is not None
        )

    # async def _resume_story(self) -> None:
    #     """
    #     Resumes the story generation process from the last saved state.
    #     """
    #     logger.info("Attempting to resume story generation from the last saved state.")
    #     latest_story_dir = get_latest_story_directory(
    #         self.config.output_artifacts.stories_root_dir
    #     )

    #     self.story_manager = InMemStoryData.load_state(directory=latest_story_dir)
    #     if self.story_manager:
    #         last_step = self.story_manager.metadata.get(
    #             "step", StoryState.USER_FACING_CHAT
    #         )
    #         if last_step is StoryState.FINAL_DOCUMENT_GENERATED:
    #             logger.info(f"Last story was completed. We will create a new story.")
    #             await self._start_new_story()
    #         else:
    #             logger.info(f"Resuming story generation from step: {last_step}")
    #         await self._process_story_creation_step(last_step)
    #     else:
    #         logger.info("No saved state found, starting a new story.")
    #         await self._start_new_story()

    async def _process_story_creation_step(
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

        # # Cancel the token refresh task when the story generation process is completed or fails
        # if self.token_refresh_task:
        #     self.token_refresh_task.cancel()

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

    def _reset(self) -> None:
        """
        Reset the variables for a fresh run.

        This method sets all internal state variables to their initial values.
        """
        try:
            # Resetting the StoryManager object to its initial state
            self.story_manager.reset()

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


    async def _user_facing_chat(self) -> None:
        """
        Start and run the WebSocket server to gather user information through a chat with the GUI.

        This method handles the user-facing chat phase, processes the completed chat,
        extracts story elements, and updates the database and knowledge base accordingly.
        """
        logger.info("Starting User-Facing chat Phase")

        self.chat_completed = False

        # Wait for chat to complete
        await self.chat_assistant.wait_for_chat_completion()

        # Process the completed chat
        await self.on_chat_completed(await self.chat_assistant._retrieve_messages())

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

    async def handle_assistant_requests(self, ai_message_for_system: WSInput) -> None:
        """
        Processes the Chat AI Assistant message.
        """
        logger.info(
            f"Processing AI message for Magic-Tales Orchestrator: {ai_message_for_system}"
        )

        if not isinstance(ai_message_for_system, WSInput):
            logger.info(
                f"AI message for Magic-Tales Orchestrator is not a WSInput: {ai_message_for_system}"
            )
            return

        command = ai_message_for_system.command

        if command == Command.CHAT_COMPLETED and not self.chat_completed:
            logger.info("Chat completed.")
            self.profile_id = ai_message_for_system.profile_id
            # Ensure there is a profile associated with the story
            if not self.profile_id:
                logging.error(
                    "Profile ID is required but not provided by the assistant at 'chat_completed' event. Asking assistant to solve"
                )
                await self._generate_and_send_update_message_for_user(
                    self.config.updates_request_prompts.no_profile_id
                )
                return

            current_profile = await self.database_manger.fetch_profile_by_id(self.profile_id)
            if not current_profile:
                logging.error(
                    "No profile found with the given ID by the assistant at 'chat_completed' event. Asking assistant to solve"
                )
                await self._generate_and_send_update_message_for_user(
                    self.config.updates_request_prompts.profile_id_does_not_exist
                )
                return

            self.chat_assistant.chat_completed_event.set()
            self.chat_completed = True
            return

        if command == Command.CONTINUE_STORY_CREATION:
            if ai_message_for_system.message == "true":
                asyncio.create_task(self._resume_story())
            else:
                asyncio.create_task(self._handle_new_tale())
            return

        if command == Command.UPDATE_PROFILE:
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.updating_profile
            )
            await self._update_profile_from_chat(ai_message_for_system)
            await self._fetch_user_data_and_update_knowledge_base()
            return

        if command == Command.NEW_PROFILE:
            await self._generate_and_send_update_message_for_user(
                self.config.updates_request_prompts.creating_new_profile
            )
            await self.database_manger.create_profile_from_chat_details(ai_message_for_system)
            await self._fetch_user_data_and_update_knowledge_base()    

    async def _update_profile_from_chat(self, ai_message_for_system: WSInput):
        """
        Updates a profile based on the AI system message.

        Args:
            ai_message_for_system (WSInput): The message from the AI system with update details.
        """
        profile_id = ai_message_for_system.profile_id
        updates = ai_message_for_system.message
        logger.info(f"profile UPDATES:\n{updates}")

        # Fetch existing profile details
        current_profile = await self.database_manger.fetch_profile_by_id(profile_id)
        existing_details = current_profile.details if current_profile else ""
        logger.info(f"current profile details:\n{existing_details}")

        # Merge updates using the Information Extractor Assistant
        merged_details = await self.merge_profile_updates(existing_details, updates)
        logger.info(f"merged profile details:\n{merged_details}")

        await self.database_manger.update_profile_record_by_id(profile_id, {"details": merged_details})

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

    async def on_chat_completed(self, messages: List[Any]) -> None:
        """
        Callback for handling the completion of the user-facing chat.

        Args:
            messages (List[ThreadMessage]): List of chat messages with user interactions.

        This method processes the chat messages, extracts relevant story elements,
        and updates the database and knowledge base.
        """
        logger.info("User-Facing chat complete. Processing chat messages.")
        await self.send_working_command_to_frontend(True)

        await self._generate_and_send_update_message_for_user(
            self.config.updates_request_prompts.chat_info_extraction
        )

        chat_string = await self._convert_chat_to_string(messages)
        updated_elements = await self._extract_chat_key_elements(chat_string)

        logger.info("Updating the database with new story elements from chat")
        await self.create_story_foundation_from_chat_elements(updated_elements)

        logger.info("Database updated with new story elements from chat")

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
                # Check if the content is of type MessageContentText and extract text
                if isinstance(content, MessageContentText):
                    chat_lines.append(f"{msg.role}: {content.text}")
                # Optionally handle MessageContentImageFile types differently, e.g., by noting an image was present
                # else if isinstance(content, MessageContentImageFile):
                #     chat_lines.append(f"{msg.role}: [Image content]")
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
            f"Identify story features from this conversation:\n{chat_string}"
        )

        story_synopsis, _ = await self.helper_assistant.request_ai_response(
            f"Extract the latest agreed upon story synopsis from this conversation:\n{chat_string}"
        )

        story_title, _ = await self.helper_assistant.request_ai_response(
            f"Extract the story title from this synopsis:\n{story_synopsis}"
        )
        extracted_info = {
            "story_features": story_features,
            "story_synopsis": story_synopsis,
            "story_title": story_title,
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
        if not self.profile_id:
            logger.error("Profile ID is missing but required for story creation.")
            raise ValueError("Profile ID is required but not provided.")

        # Fetch the current profile using the DataManager class
        current_profile = await self.database_manger.fetch_profile_by_id(self.profile_id)
        if not current_profile:
            logger.error(f"No profile found for ID: {self.profile_id}")
            raise ValueError("No profile found with the given ID. Cannot create a story.")

        # Extract story elements from updates
        title = updates.get("story_title")
        features = updates.get("story_features")
        synopsis = updates.get("story_synopsis")
        
        # Validate extracted story elements
        if not all([title, features, synopsis]):
            logger.error("Missing one or more mandatory story elements (title, features, synopsis)")
            raise ValueError(f"fMissing mandatory story elements.\nTitle: {title}\nFeatures:{features}\nSynopsis:{synopsis}")
        
        # Define paths for story and image storage based on user-specific directories
        stories_root_dir = os.path.join(self.config.output_artifacts.stories_root_dir, f"user_{self.user_id}")
    
        story_folder, images_subfolder = create_new_story_directory(
            stories_root_dir, subfolders=self.subfolders
        )

        # Create the story using the StoryManager class
        try:
            await self.story_manager.create_story(
                profile_id=self.profile_id,
                session_id=self.session.identity_key,  # Session identity key needs to be accurately sourced
                title=title,
                features=features,
                synopsis=synopsis,
                story_folder=story_folder,
                images_subfolder=images_subfolder,
            )
            logger.info(f"Story foundation created successfully with ID: {self.story_manager.story.id}")
        except Exception as e:
            logger.error(f"Failed to create story foundation: {e}")
            raise Exception("Failed to create the story foundation due to an unexpected error.") from traceback.format_exc()


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

        Generates:
            List[Dict[str, str]]: A list of dictionaries, each containing the chapter title and content.

        Resturns:
            None
        """
        await self._generate_and_send_update_message_for_user(
            self.config.updates_request_prompts.generating_story
        )

        # Initialize a list to hold the chapters
        chapters = []

        # Step 1: Determine Chapter Count
        # TODO: Make this chapter count more robust
        num_chapters = await self._determine_chapter_count(
            self.story_manager.story.features
        )
        num_chapters = max(1, num_chapters)
        logger.info(f"Number of chapters: {num_chapters}")

        # Step 2: Initialize chapter content
        previous_chapter_content = ""

        # Step 3: Chapter Generation
        chapters_folder = os.path.join(
            self.story_manager.story.story_folder, self.subfolders["chapters"]
        )
        for chapter_number in range(1, num_chapters + 1):
            message = f"Generating chapter: {chapter_number} of {num_chapters}"
            logger.info(message)

            update = await self.create_progress_update(
                story_state=str(StoryState.STORY_GENERATION),
                status=ResponseStatus.STARTED,
                progress_percent=(chapter_number / num_chapters) * 100,
                message=message,
            )
            await self.send_message_to_frontend(update)

            chapter_title, chapter_content = await self._generate_chapter(
                chapters_folder=chapters_folder,
                chapter_number=chapter_number,
                total_number_chapters=num_chapters,
                previous_chapter_content=previous_chapter_content,
            )
            # Add the generated chapter to the chapters list
            chapters.append({"title": chapter_title, "content": chapter_content})
            previous_chapter_content = chapter_content

        self.story_manager.in_mem_story_data.chapters = chapters
        return

    async def _determine_chapter_count(self, story_features: str) -> Union[int, None]:
        """
        Determine the number of chapters in the story based on given story features.

        Args:
            story_features (str): The features of the story.

        Returns:
            int: The determined number of chapters. Returns None if unable to determine.
        """
        logger.info("Determining the number of chapters in the story.")

        # Request the number of chapters from the agent
        message = f"Given the following information:\n\nStory main features: {story_features}.\n\nPlease,infer the number of chapters of the story we want to create. Respond just with a numerical value. For example: 1, 5, 10... If unknown, default to 1"
        response, _ = await self.helper_assistant.request_ai_response(message)
        logger.info(f"{response}")

        # try to extract numerical values
        try:
            return int(response)
        except ValueError:
            pass

        # First, try to extract numerical values
        match = re.search(r"The number of chapters should be (\d+)", response)
        if match:
            return int(match.group(1))

        # Second, try to convert spelled-out numbers to integers
        match = re.search(r"The number of chapters should be ([a-zA-Z]+)", response)
        if match:
            try:
                return w2n.word_to_num(match.group(1))
            except ValueError:
                pass

        # Third, try to handle a range (e.g., "between 5 to 7") and take the lower limit
        match = re.search(r"between (\d+) to (\d+)", response)
        if match:
            return int(match.group(1))

        # Third, try to handle a range (e.g., "between 5 to 7") and take the lower limit
        match = re.search(r"between (\d+) and (\d+)", response)
        if match:
            return int(match.group(1))

        # If no suitable format found, default to None (or 1, depending on your preference)
        logger.warning("Unable to determine the number of chapters. Defaulting to 1.")
        return 1

    async def _generate_chapter(
        self,
        chapters_folder: str,
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
        self.chapter_generation_mechanism = ChapterGenerationMechanism(
            config=self.config,
            story_blueprint=self.story_manager.get_story_blueprint(),
            previous_chapter_content=previous_chapter_content,
        )
        best_chapter = self.chapter_generation_mechanism.create_chapter(
            chapters_folder=chapters_folder,
            chapter_number=chapter_number,
            total_number_chapters=total_number_chapters,
        )
        chapter_content = best_chapter.get("chapter_generator_response_dict", {}).get(
            "content", ""
        )
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

    async def _generate_image_prompts(self) -> None:
        """Generate image prompts for all chapters in the story."""

        message = f"Generating image prompts for all chapters."
        logger.info(message)
        update = await self.create_progress_update(
            story_state=str(StoryState.IMAGE_PROMPT_GENERATION),
            status=ResponseStatus.STARTED,
            progress_percent=0,
            message=message,
        )
        await self.send_message_to_frontend(update)
        image_prompt_generation_mechanism = ImagePromptGenerationMechanism(
            config=self.config
        )
        all_chapter_prompts_and_new_chapters_annotated = (
            image_prompt_generation_mechanism.generate_image_prompts_for_all_chapters(
                chapters=self.story_manager.in_mem_story_data.chapters
            )
        )
        await self._extract_post_processed_chapters(
            all_chapter_prompts_and_new_chapters_annotated
        )
        logger.info(
            f"Image Prompts + processed chapters created"
        )  # {self.story_manager.in_mem_story_data.image_prompts}")

        await self.send_working_command_to_frontend(False)

    async def _extract_post_processed_chapters(
        self, all_chapter_prompts: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Save the post-processed chapters, which include image annotations, into the InMemStoryData structure.

        Args:
            all_chapter_prompts: A dictionary containing image prompt data for each chapter.

        Returns:
            None
        """
        post_processed_chapters = []
        image_prompt_messages = []
        image_prompts = []

        for chapter_number, chapter_data in all_chapter_prompts.items():
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

        all_chapter_prompts = self.story_manager.in_mem_story_data.image_prompts
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
            for chapter_number, chapter_prompts in enumerate(all_chapter_prompts, 1)
            for image_prompt_index, image_prompt in enumerate(chapter_prompts)
        ]

        for chapter_number, image_prompt, image_prompt_index in flat_image_prompts:
            filename = f"Chapter_{chapter_number}_Image_{image_prompt_index}.png"
            message = f"Generating image for Chapter {chapter_number}, Image {image_prompt_index}"
            logger.info(message)
            update = await self.create_progress_update(
                story_state=str(StoryState.IMAGE_GENERATION),
                status=ResponseStatus.STARTED,
                progress_percent=0,
                message=message,
            )
            await self.send_message_to_frontend(update)

            try:
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
                    update = await self.create_progress_update(
                        story_state=str(StoryState.IMAGE_GENERATION),
                        status=ResponseStatus.STARTED,
                        progress_percent=0,
                        message=message,
                    )
                    await self.send_message_to_frontend(update)
            except Exception as e:
                message = f"An error occurred while processing Chapter {chapter_number}, Image {image_prompt_index}."
                logger.error(f"{message}. Details:\n{traceback.format_exc()}")
                update = await self.create_progress_update(
                    story_state=str(StoryState.IMAGE_GENERATION),
                    status=ResponseStatus.FAILED,
                    progress_percent=0,
                    message=message,
                )
                await self.send_message_to_frontend(update)
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
        message = f"Starting document generation..."
        logger.info(message)

        try:

            update = await self.create_progress_update(
                story_state=str(StoryState.DOCUMENT_GENERATION),
                status=ResponseStatus.STARTED,
                progress_percent=0,
                message=message,
            )
            await self.send_message_to_frontend(update)

            story_document = StoryDocument(
                title=self.story_manager.story.title,
                chapters=self.story_manager.in_mem_story_data.post_processed_chapters,
                image_filenames=self.story_manager.in_mem_story_data.image_filenames,
                story_folder=self.story_manager.story.story_folder,
                images_subfolder=self.story_manager.story.images_subfolder,
            )

            story_document.create_document()
            doc_filepath = story_document.save_document("story.docx")
            final_pdf_file_path = story_document.convert_docx_to_pdf(doc_filepath)

            logger.info(f"Story document saved in: {final_pdf_file_path}")

            message = "We really hope you enjoy this story!"
            progress_percent = 100

        except Exception as e:
            message = "Failed to generate final document"
            progress_percent = 0
            logger.error(message, exc_info=True)
            raise (f"{message}: {e}\n{traceback.format_exc()}")
        finally:
            http_base_url = "http://localhost:8000"
            update = await self.create_progress_update(
                command=Command.PROCESS_COMPLETED,
                story_state=str(StoryState.DOCUMENT_GENERATION),
                files=[http_base_url + final_pdf_file_path],
                data={"story_id": self.story_manager.story.id},
                progress_percent=progress_percent,
                message=message,
            )
            await self.send_message_to_frontend(update)

            await self.send_working_command_to_frontend(False)
