import os
import copy
import traceback
import asyncio
import json
from typing import List, Tuple, Optional, Dict, Union, Any, NoReturn
from omegaconf import DictConfig
from hydra import compose
from hydra.utils import instantiate

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

from services.openai_assistants.assistant_input.assistant_input import (
    AssistantInput,
    Source,
)
from services.openai_assistants.chat_assistant.chat_assistant import ChatAssistant
from services.openai_assistants.chat_assistant.chat_assistant_input import (
    ChatAssistantInput,
)
from services.openai_assistants.chat_assistant.chat_assistant_response import (
    ChatAssistantResponse,
)
from services.openai_assistants.prompt_utils import (
    async_load_prompt_template_from_file,
)
from services.message_sender import MessageSender
from services.openai_assistants.helper_assistant.helper_assistant import HelperAssistant
from services.openai_assistants.helper_assistant.helper_assistant_input import (
    HelperAssistantInput,
)
from services.openai_assistants.helper_assistant.helper_assistant_response import (
    HelperAssistantResponse,
)
from services.openai_assistants.supervisor_assistant.supervisor_assistant import (
    SupervisorAssistant,
)
from services.openai_assistants.supervisor_assistant.supervisor_assistant_input import (
    SupervisorAssistantInput,
)
from services.openai_assistants.supervisor_assistant.supervisor_assistant_response import (
    SupervisorAssistantResponse,
)
from services.utils.file_utils import (
    create_new_story_directory,
    get_latest_story_directory,
    convert_user_info_to_json_files,
    convert_user_info_to_1_json_file,
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
from services.custom_exceptions.custom_exceptions import NotADictionaryError

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


from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)

# TODO: do not overwrite the value of the environment variable if it exists
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

        # Configuration
        self.config = copy.deepcopy(config)

        # Initialize the Chat Assistant
        self.chat_assistant = ChatAssistant(config=self.config.chat_assistant)

        # Initialize the Helper Assistant
        self.helper_assistant = HelperAssistant(config=config.helper_assistant)

        # Initialize the Supervisor Assistant
        self.supervisor_assistant = SupervisorAssistant(
            config=self.config.supervisor_assistant
        )

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

        self.waiting_for_response_to_continue_where_we_left_off = False

        self.frontend_command_handlers = {
            Command.NEW_TALE: self.handle_command_new_tale,
            Command.SPIN_OFF: self.handle_command_spin_off,
            Command.UPDATE_PROFILE: self.handle_command_update_profile,
            Command.CONVERSATION_RECOVERY: self.handle_command_conversation_recovery,
            Command.LINK_USER_WITH_CONVERSATIONS: self.handle_command_link_user_with_conversations,
            Command.USER_MESSAGE: self._handle_communication_with_assistant,
            # Command.USER_LOGGED_IN: self.handle_command_new_tale,
        }

        self.user_language = None

        self.chat_supervision_required = None

        logger.info("MagicTales initialized.")    

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

        if self.user_id != coming_user_id and coming_user_id is not None:
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
        supervisor_id = self.user_dict.get("supervisor_id", None)

        self.user_language = self.user_dict.get("language", "ENG")

        logger.info(f"User current Assistant id on DB: {assistant_id}")
        logger.info(f"User current Helper id on DB: {helper_id}")
        logger.info(f"User current Supervisor id on DB: {supervisor_id}")

        files_paths = await self._fetch_user_data_and_update_knowledge_base()
        await self._initialize_assistants(
            assistant_id, helper_id, supervisor_id, files_paths
        )

        user_message = await self._generate_starting_message(try_mode=try_mode)
        await self._generate_system_request_to_update_user(user_message)

        await self.story_manager.refresh()

    async def process_command(self, frontend_request: WSInput):
        """
        Process the incoming command using the appropriate handler from the self.frontend_command_handlers dictionary.

        Args:
            frontend_request (WSInput): The request from the frontend containing the command to process.
        """
        handler = self.frontend_command_handlers.get(frontend_request.command)
        if handler:
            logger.warning(f"FrontEnd Command received: {frontend_request.command}")
            await handler(frontend_request)
        else:
            logger.warning(f"Unknown FrontEnd Command: {frontend_request.command}")

    async def _initialize_assistants(
        self,
        assistant_id: Optional[str],
        helper_id: Optional[str],
        supervisor_id: Optional[str],
        files_paths: Optional[List[str]],
    ) -> None:
        """
        Initialize chat and helper assistants.

        Args:
            assistant_id (Optional[str]): The ID of the chat assistant.
            helper_id (Optional[str]): The ID of the helper assistant.
            supervisor_id (Optional[str]): The ID of the supervisor assistant.
        """
        # CHAT ASSISTANT
        try:
            await self.chat_assistant.start_assistant(assistant_id, files_paths)
        except ValueError as e:
            logger.error(f"Error initializing the Chat assistant: {e}")
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

        # HELPER ASSISTANT
        try:
            await self.helper_assistant.start_assistant(files_paths)
        except ValueError as e:
            logger.error(f"Error initializing the Helper assistant: {e}")
            await self._send_error_message(
                "We are currently experiencing technical difficulties. Please try again later."
            )
            return        

        # SUPERVISOR ASSISTANT
        try:
            await self.supervisor_assistant.start_assistant(None, files_paths)
        except ValueError as e:
            logger.error(f"Error initializing the Supervisor assistant: {e}")
            await self._send_error_message(
                "We are currently experiencing technical difficulties. Please try again later."
            )
            return

        if supervisor_id != self.supervisor_assistant.openai_assistant.id:
            logger.warn(
                f"User {self.user_id} had Supervisor assistant {supervisor_id}, which was not found. It's going to be updated. This should only happen with brand new users!"
            )
            await self.database_manager.update_user_record_by_id(
                self.user_id,
                {"supervisor_id": self.supervisor_assistant.openai_assistant.id},
            )

    async def handle_command_new_tale(self, frontend_request: WSInput):
        if await self.last_story_finished_correctly(self.user_id):
            asyncio.create_task(self._handle_new_tale())

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

        await self.last_story_finished_correctly(self.user_id)
        # await self._generate_system_request_to_update_user(
        #     self.config.updates_request_prompts.coversation_recovery,
        #     conversation=conversation_dicts,
        # )

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
        self.chat_supervision_required = True
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
                self.waiting_for_response_to_continue_where_we_left_off = False
                return True
            else:
                logger.info(
                    f"Last story completed step was: {StoryState(self.latest_story.last_successful_step).name}"
                )
                self.waiting_for_response_to_continue_where_we_left_off = True
                # Process to resume the story
                await self._generate_system_request_to_update_user(
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

    async def supervisor_chat_assistant_message(
        self, num_messages: int
    ) -> SupervisorAssistantResponse:
        """
        Sends the latest messages to the supervisor assistant for analysis and intervention.

        Args:
            num_messages (int): The number of latest messages to send to the supervisor.

        Returns:
            SupervisorAssistantResponse: The response from the supervisor assistant.
        """
        all_messages = self.chat_assistant.latest_messages_data
        latest_messages = all_messages[:num_messages]

        formatted_messages = self._format_messages(latest_messages)
        message_json = json.dumps(formatted_messages)

        message_for_supervisor = SupervisorAssistantInput(
            message=message_json, source=Source.USER
        )

        ai_supervisor_response: SupervisorAssistantResponse = (
            await self.supervisor_assistant.request_ai_response(message_for_supervisor)
        )

        return ai_supervisor_response

    def _format_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """
        Extracts relevant information from Message objects.

        Args:
            messages (List[Any]): List of Message objects.

        Returns:
            List[Dict[str, str]]: Formatted messages.
        """
        return [
            {
                "role": message.role,
                "content": message.content[0].text.value if message.content else "",
            }
            for message in messages
        ]

    async def _handle_communication_with_assistant(
        self, request: WSInput, source: Source = Source.USER
    ) -> None:
        """
        Handle a user message by sending it to the chat assistant and processing the response.

        Args:
            request (WSInput): The message to be sent to the Assistant.
            source (Source): whether the message is from the user or from the system

        Returns:
            None
        """
        await self.send_working_command_to_frontend(True)

        message_for_assistant = ChatAssistantInput(
            message=request.message, source=source
        )

        # Generate AI response for the user message
        chat_assistant_response: ChatAssistantResponse = (
            await self.chat_assistant.request_ai_response(message_for_assistant)
        )
        ai_message_for_user = await chat_assistant_response.get_message_for_user()
        ai_message_for_system = await chat_assistant_response.get_message_for_system()

        logger.info(f"Chat Assistant response to User: {ai_message_for_user}")
        logger.info(f"Chat Assistant response to Sys: {ai_message_for_system}")

        ##################### SUPERVISION ####################
        # Supervise the Chat Assistant message before doing anything
        if self.chat_supervision_required:
            ai_supervisor_response = await self.supervisor_chat_assistant_message(
                num_messages=self.config.supervisor_assistant.num_last_messages_to_use
            )
            if await ai_supervisor_response.get_intervention_needed():
                self.supervisor_assistant.interventions_count += 1
                if (
                    self.supervisor_assistant.interventions_count
                    <= self.supervisor_assistant.interventions_count_limit
                ):
                    logger.warning(f"Supervisor: Intervention_needed!!")
                    intervention_message = (
                        await ai_supervisor_response.get_message_for_user()
                    )
                    logger.info(f"Intervetion message:{intervention_message}")
                    await self._generate_system_request_to_update_user(
                        intervention_message
                    )
                    return

            self.supervisor_assistant.interventions_count = 0

        # Construct a response to be sent by the AI Core Interface Layer to the Front End, so the user can see it
        chat_assistant_response_for_frontend = WSOutput(
            command=Command.MESSAGE_FOR_HUMAN,
            message=ai_message_for_user,  # AI-generated response to be displayed for the user
            token=self.new_token,
            working=False,  # Indicates that the response is ready
        )
        await self.send_message_to_frontend(chat_assistant_response_for_frontend)
        await self.send_working_command_to_frontend(False)

        if source == Source.USER:
            self.chat_supervision_required = True
            self.user_language = (
                await chat_assistant_response.get_user_language() or "ENG"
            )
            logger.info(f"AI detected user language: {self.user_language}")

        # If Chat Assitant Didn't need any intervention we continue here.
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

        try:
            # Retrieve the story details from the database
            story = await self.database_manager.fetch_story_by_id(request.story_id)
            if not story:
                logger.error(f"No story found with ID: {request.story_id}")
                # await self._send_error_message_to_chat_assistant(
                #     "The specified story does not exist. Please provide a valid story ID."
                # )
                return

            spin_off_story = story.to_dict()

            await self._generate_system_request_to_update_user(
                self.config.updates_request_prompts.spin_off_clicked,
                story=spin_off_story,
            )

        except Exception as e:
            logger.exception(
                f"An error occurred while handling the spin-off tale: {str(e)}"
            )
            await self._send_error_message_to_chat_assistant(
                "An unexpected error occurred while processing the spin-off tale request. Please try again later."
            )

    async def _handle_new_tale(self) -> None:
        """
        Handle the 'new-tale' command by retrieving user information from the database.

        Returns:
            None.
        """
        logger.info("Starting story generation from scratch.")
        await self._reset()
        await self._generate_system_request_to_update_user(
            self.config.updates_request_prompts.new_tale_clicked
        )

    async def _generate_system_request_to_update_user(
        self, request_message: str, **replacements: str
    ) -> None:
        """
        Sends an update message to the user.

        Args:
            request_message (str): This is the message we are sending the Assistant to generate an amazing update for the user.
            **replacements: Keyword arguments representing the key-value pairs for replacements.
                The keys are the expressions to be replaced, and the values are the corresponding replacement strings.

        Returns:
            None.
        """
        # By Default
        request_message = request_message.replace("{user_info}", f"{self.user_dict}")

        for key, value in replacements.items():
            request_message = request_message.replace(f"{{{key}}}", str(value))

        request_message = (
            request_message
            + " To respond, use strictly this language used by the user:"
            + self.user_language
        )
        request_message = WSInput(
            command=Command.USER_MESSAGE, token=self.new_token, message=request_message
        )

        await self._handle_communication_with_assistant(
            request=request_message, source=Source.SYSTEM
        )

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

        # files_paths = await convert_user_info_to_json_files(
        #     current_knowledge_base,
        #     self.config.output_artifacts.knowledge_base_root_dir,
        # )

        files_paths = await convert_user_info_to_1_json_file(
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
        self.chat_supervision_required = False

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
            self.chat_supervision_required = True
            logger.info(
                f"Reset executed. chat_supervision_required:{self.chat_supervision_required}"
            )

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

    async def _generate_starting_message(self, try_mode: bool = False) -> str:
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
            await self._generate_system_request_to_update_user(
                self.config.updates_request_prompts.no_command
            )
            return

        if command == Command.START_STORY_GENERATION:
            await self._handle_start_story_generation(ai_message_for_system)
            return

        if (
            command == Command.CONTINUE_UNFINISHED_STORY_GENERATION
            and self.waiting_for_response_to_continue_where_we_left_off
        ):
            await self._handle_continue_where_we_left_off_response(
                ai_message_for_system
            )
            return

        if command == Command.UPDATE_PROFILE:
            if await self._update_profile_from_chat(ai_message_for_system):
                files_paths = await self._fetch_user_data_and_update_knowledge_base()
                await self.chat_assistant.update_assistant(files_paths=files_paths)
                await self._generate_system_request_to_update_user(
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
            # else:
            #     await self._generate_system_request_to_update_user(
            #         self.config.updates_request_prompts.failed_profile_update
            #     )

            return

        if command == Command.NEW_PROFILE:
            if await self._create_new_profile_from_chat(ai_message_for_system):
                files_paths = await self._fetch_user_data_and_update_knowledge_base()
                await self.chat_assistant.update_assistant(files_paths=files_paths)
                await self._generate_system_request_to_update_user(
                    self.config.updates_request_prompts.profile_created
                )
                await self.story_manager.refresh()
            # else:
            #     await self._generate_system_request_to_update_user(
            #         self.config.updates_request_prompts.failed_profile_creation
            #     )
            return

    async def _check_for_correct_keys_within_command(
        self, command: str, required_keys: set, ai_message_for_system: dict
    ) -> bool:
        """
        Checks if the AI message contains all required information to start story generation.

        Args:
            command (str): The command being executed.
            required_keys (set): The set of required keys for the command.
            ai_message_for_system (dict): The message_for_system dictionary from the AI response.

        Returns:
            bool: True if all required information is present, False otherwise.
        """
        missing_keys = required_keys - set(ai_message_for_system.keys())

        if missing_keys:
            logger.error(f"Missing required keys in message_for_system: {missing_keys}")

            update_message = (
                self.config.updates_request_prompts.missing_keys_in_command.format(
                    missing_keys=", ".join(missing_keys), command=command
                )
            )
            await self._generate_system_request_to_update_user(update_message)

            return False

        return True

    async def _check_profile_exists(
        self,
        ai_message_for_system: dict,
        profile_fields_mapping: Optional[dict] = None,
    ) -> bool:
        """
        Checks if the selected profile exists in the database.

        Args:
            command (str): The command being executed.
            ai_message_for_system (dict): The message_for_system dictionary from the AI response.
            profile_fields_mapping (Optional[dict]): The mapping of profile fields to their corresponding names in the AI message.

        Returns:
            bool: True if the profile exists, False otherwise.
        """
        profile_fields_mapping = profile_fields_mapping or {
            # "id": "profile_id",
            "name": "name",
            "age": "age",
            "user_id": "user_id",
        }

        profile_data = {
            field: ai_message_for_system[mapped_name]
            for field, mapped_name in profile_fields_mapping.items()
        }

        current_profile = await self.database_manager.fetch_profile_by_fields(
            **profile_data
        )

        if not current_profile:
            logger.warn("No profile found!")
            return False

        ai_message_for_system["profile_id"] = current_profile.id
        return True

    async def _handle_start_story_generation(self, ai_message_for_system: dict) -> None:
        """
        Handles the start_story_generation command from the AI assistant.

        Args:
            ai_message_for_system (dict): The message_for_system dictionary from the AI response.
        """
        logger.info("start_story_generation request.")
        required_keys = {"name", "age", "user_id"}  # "profile_id",

        if not await self._check_for_correct_keys_within_command(
            Command.START_STORY_GENERATION, required_keys, ai_message_for_system
        ):
            return

        if not await self._check_profile_exists(ai_message_for_system):
            await self._generate_system_request_to_update_user(
                self.config.updates_request_prompts.profile_selected_does_not_exist.format(
                    command=Command.START_STORY_GENERATION
                )
            )
            return

        logger.info("Updating the database with new story elements from chat")
        await self._generate_system_request_to_update_user(
            self.config.updates_request_prompts.chat_info_extraction
        )

        await self.send_working_command_to_frontend(True)
        chat_messages = await self.chat_assistant._retrieve_messages()
        story_info_dict = await self._extract_chat_key_elements(chat_messages)
        story_info_dict["profile_id"] = ai_message_for_system["profile_id"]

        await self.create_story_foundation_from_chat_elements(story_info_dict)
        logger.info("Database updated with new story elements from chat")

        await self._process_story_generation_step(StoryState.STORY_GENERATION)
        await self.send_working_command_to_frontend(False)

    async def _handle_continue_where_we_left_off_response(
        self, ai_message_for_system: dict
    ) -> None:
        """
        Handles whether the user want to continue an unfinished story where we left it off, or not.
        """
        logger.info("handling continue_where_we_left_off response")
        required_keys = {"continue_where_we_left_off"}

        if not await self._check_for_correct_keys_within_command(
            Command.CONTINUE_UNFINISHED_STORY_GENERATION,
            required_keys,
            ai_message_for_system,
        ):
            return

        continue_where_we_left_off = ai_message_for_system.get(
            "continue_where_we_left_off"
        )

        logger.info(
            f"'Continue where we left off' RESPOSE received:{continue_where_we_left_off}"
        )

        # Ensure that we got a proper response
        if continue_where_we_left_off.lower() not in ("true", "false"):
            logging.error(
                "'Continue where we left off' command by the assistant had no response. Asking assistant to solve it"
            )
            await self._generate_system_request_to_update_user(
                self.config.updates_request_prompts.continue_where_we_left_off_response_incorrect
            )
            return

        self.waiting_for_response_to_continue_where_we_left_off = False
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
        required_keys = {
            "profile_id",
            "current_name",
            "current_age",
            "user_id",
            "updated_name",
            "updated_age",
            "updated_details",
        }
        profile_fields_mapping = {
            "id": "profile_id",
            "name": "current_name",
            "age": "current_age",
            "user_id": "user_id",
        }
        if not await self._check_for_correct_keys_within_command(
            Command.UPDATE_PROFILE, required_keys, ai_message_for_system
        ):
            return False

        if not await self._check_profile_exists(
            ai_message_for_system, profile_fields_mapping
        ):
            await self._generate_system_request_to_update_user(
                self.config.updates_request_prompts.profile_selected_does_not_exist.format(
                    command=Command.UPDATE_PROFILE
                )
            )
            return False

        profile_id = ai_message_for_system["profile_id"]

        # Ensure there is at least something to update
        updated_name = ai_message_for_system["updated_name"]
        updated_age = ai_message_for_system["updated_age"]
        updated_details = ai_message_for_system["updated_details"]

        has_changed = updated_name or updated_age or updated_details
        if not has_changed:
            logging.error(
                "At least one of the following name, age or details are required for profile update, but one or more were NOT provided by the assistant. Asking assistant to solve"
            )
            await self._generate_system_request_to_update_user(
                self.config.updates_request_prompts.no_info_at_update_profile
            )
            return

        logger.info(
            f"profile UPDATES requested:\nName:{updated_name}\nAge:{updated_age}\nDetails:{updated_details}"
        )

        try:
            update_dict = {}
            if updated_name:
                update_dict["name"] = updated_name
            if updated_age:
                update_dict["age"] = updated_age
            if updated_details:
                update_dict["details"] = updated_details

            # Only perform the update if there are fields to update
            if update_dict:
                await self.database_manager.update_profile_record_by_id(
                    profile_id, update_dict
                )
                await self.story_manager.refresh()
                return True
            else:
                logger.warning("No fields to update. Skipping update.")
                return False

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
        required_keys = {
            "name",
            "age",
            "details",
        }
        profile_fields_mapping = {
            "name": "name",
            "age": "age",
            "user_id": "user_id",
        }

        if not await self._check_for_correct_keys_within_command(
            Command.UPDATE_PROFILE, required_keys, ai_message_for_system
        ):
            return

        name = ai_message_for_system.get("name")
        age = ai_message_for_system.get("age")
        details = ai_message_for_system.get("details")
        user_id = self.user_id

        new_profile = {
            "name": name,
            "age": age,
            "details": details,
            "user_id": user_id,
        }

        # Ensure there ALL Fields are there, not just some
        if not name or not age or not details:
            logging.error(
                "Name, age and details are ALL required to create a new profile, but none were provided by the assistant. Asking assistant to solve"
            )
            await self._generate_system_request_to_update_user(
                self.config.updates_request_prompts.no_info_at_create_profile
            )
            return False

        if await self._check_profile_exists(new_profile, profile_fields_mapping):
            logger.error(
                f"Profile Already Exist:\nName: {name}\nAge: {age}\nUser_id:{user_id}"
            )
            await self._generate_system_request_to_update_user(
                self.config.updates_request_prompts.new_profile_already_exists()
            )
            return False

        try:
            await self.database_manager.create_profile_from_chat_details(new_profile)
            await self.story_manager.refresh()
            logger.info(
                f"NEW profile created:\nName:{name}\nAge:{age}\nDetails:{details}\nuser_id: {self.user_id}"
            )

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
        chat_string = HelperAssistantInput(
            message=f"Existing details: {existing_details}\nUpdates: {updates}\n"
            f"Merge these updates into the existing details. "
            + "To respond, use strictly this language used by the user:"
            + self.user_language,
            source=Source.USER,
        )

        # Use the Information Extractor to merge the details.
        ai_response: HelperAssistantResponse = (
            await self.helper_assistant.request_ai_response(chat_string)
        )
        merged_details = await ai_response.get_message_for_user()
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
        if not self.config.helper_assistant.story_features_extraction_prompt_path:
            raise ValueError(
                "config.helper_assistant.story_features_extraction_prompt_path is required for feature extraction"
            )

        # 1. EXTRACT THE STORY FEATURES
        story_features_extraction_prompt = await async_load_prompt_template_from_file(
            self.config.helper_assistant.story_features_extraction_prompt_path
        )
        story_features_extraction_prompt = story_features_extraction_prompt.replace(
            "{chat_string}", f"{chat_string}"
        )
        story_features_extraction_prompt = story_features_extraction_prompt.replace(
            "{user_language}", f"{self.user_language}"
        )
        ai_response: HelperAssistantResponse = (
            await self.helper_assistant.request_ai_response(
                HelperAssistantInput(
                    message=story_features_extraction_prompt,
                    source=Source.USER,
                )
            )
        )
        story_features = await ai_response.get_message_for_user()

        # 2. EXTRACT THE SYNOPSIS
        synopsis_extraction_prompt = await async_load_prompt_template_from_file(
            self.config.helper_assistant.synopsis_extraction_prompt_path
        )
        synopsis_extraction_prompt = synopsis_extraction_prompt.replace(
            "{chat_string}", f"{chat_string}"
        )
        synopsis_extraction_prompt = synopsis_extraction_prompt.replace(
            "{user_language}", f"{self.user_language}"
        )

        ai_response: HelperAssistantResponse = (
            await self.helper_assistant.request_ai_response(
                HelperAssistantInput(
                    message=synopsis_extraction_prompt,
                    source=Source.USER,
                )
            )
        )
        story_synopsis = await ai_response.get_message_for_user()

        ai_response: HelperAssistantResponse = (
            await self.helper_assistant.request_ai_response(
                HelperAssistantInput(
                    message=f"Extract only the story title from this synopsis without any highligthing character such * or #:\n{story_synopsis}."
                    + "Use strictly the language used by the user:"
                    + self.user_language
                    + "\n\nThe title is:",
                    source=Source.USER,
                )
            )
        )
        story_title = await ai_response.get_message_for_user()

        ai_response: HelperAssistantResponse = (
            await self.helper_assistant.request_ai_response(
                HelperAssistantInput(
                    message=f"Extract only the number of chapters, as a single integer, from this synopsis and story features gathred from a conversation. If unknown, deduce what would be an ideal number of chapters between 1 and {self.config.output_artifacts.max_num_chapters} for this story to unfold fully:\nSynopsis:{story_synopsis}\nStory Features:{story_features}."
                    + "Use strictly the language used by the user:"
                    + self.user_language
                    + "\n\nThe number of chapters is (only an integer, no str allowed):",
                    source=Source.USER,
                )
            )
        )
        num_chapters = await ai_response.get_message_for_user()

        ## TODO: This is temporaly to perform tests. ELIMINATE BEFORE DEPLOY
        num_chapters = 2

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

        # Get the environment variables
        static_folder = os.environ.get("STATIC_FOLDER")

        stories_root_dir = os.path.join(
            static_folder,
            self.config.output_artifacts.stories_root_dir,
            f"user_{self.user_id}",
        )
        logging.warning(f"Stories root folder:{stories_root_dir}")

        story_folder, images_subfolder = create_new_story_directory(
            stories_root_dir, subfolders=self.subfolders
        )
        logging.warning(f"Story folder:{story_folder}")
        logging.warning(f"Image folder:{images_subfolder}")

        # Create the story using the StoryManager class
        try:
            await self.story_manager.create_story(
                profile_id=profile_id,
                ws_session_uid=self.websocket.uid,  # Websocket session UID associated with the story. (self.session.identity_key returns a reference to the method, not a valid attribute)
                title=title,
                features=features,
                synopsis=synopsis,
                story_folder=story_folder,
                images_subfolder=images_subfolder,
            )
            self.story_manager.in_mem_story_data.num_chapters = max(
                int(num_chapters), 1
            )

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
        message = AssistantInput(
            message=f"Given the following piece of information:\n\n1) Chapter: {chapter}.\n\nThe best possible title for this chapter is (just respond with the title, nothing else):",
            source=Source.USER,
        )

        ai_response: HelperAssistantResponse = (
            await self.helper_assistant.request_ai_response(message)
        )
        chapter_title = await ai_response.get_message_for_user()

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
            await self._generate_system_request_to_update_user(
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
            await self._generate_system_request_to_update_user(message)

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
        await self._generate_system_request_to_update_user(message)

        self.image_prompt_generation_mechanism = ImagePromptGenerationMechanism(
            config=self.config
        )

        # Generate cover image prompt
        cover_prompt = await self._generate_cover_image_prompt()

        # Generate chapter image prompts
        all_chapter_prompts_and_new_chapters_annotated = (
            await self._generate_image_prompts_for_all_chapters(
                chapters=self.story_manager.in_mem_story_data.chapters
            )
        )

        # Combine cover and chapter prompts
        all_chapter_prompts_and_new_chapters_annotated_plus_cover_prompt = {
            **cover_prompt,
            **all_chapter_prompts_and_new_chapters_annotated,
        }

        await self._extract_post_processed_chapters(
            all_chapter_prompts_and_new_chapters_annotated_plus_cover_prompt
        )
        logger.info(f"Image Prompts + processed chapters created")

    async def _generate_cover_image_prompt(self) -> Dict[int, Dict[str, Any]]:
        """Generate a single image prompt for the story cover."""
        logger.info("Generating image prompt for the story cover.")
        try:
            await self.story_manager.refresh()
            synopsis = self.story_manager.story.synopsis
            cover_prompt_data = self.image_prompt_generation_mechanism._generate_image_prompts_per_chapter(
                0, synopsis, is_cover=True
            )

            if cover_prompt_data.get("image_prompt_generator_success"):
                return {-1: {"title": "Cover", "image_prompt_data": cover_prompt_data}}
            else:
                logger.warning(
                    "Failed to generate image prompt for the cover. Skipping."
                )
                return {}
        except Exception as e:
            logger.exception(
                f"Exception while generating image prompt for the cover: {e}"
            )
            return {}

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
            chapter_title = chapter_data.get(
                "title",
                "Cover" if chapter_number == -1 else f"Chapter {chapter_number + 1}",
            )
            chapter_content = chapter_data["image_prompt_data"][
                "image_prompt_response_content_dict"
            ]["annotated_chapter"]

            image_prompt = chapter_data["image_prompt_data"][
                "image_prompt_response_content_dict"
            ]["image_prompts"]

            prompt_messages = chapter_data["image_prompt_data"][
                "image_prompt_generator_prompt_messages"
            ]

            if chapter_number >= 0:
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
        await self._generate_system_request_to_update_user(message)

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
            for chapter_number, chapter_prompts in enumerate(all_image_prompts, 0)
            for image_prompt_index, image_prompt in enumerate(chapter_prompts)
        ]

        image_generation_count = 0
        for chapter_number, image_prompt, image_prompt_index in flat_image_prompts:
            if chapter_number == 0 and image_prompt_index == 0:
                filename = "cover.png"
                message = "Update the user in just a few words that we are generating the Cover illustration for the story"
            else:
                filename = f"Chapter_{chapter_number}_Image_{image_prompt_index}.png"
                message = f"Update the user in just a few words that we are generating chapter {chapter_number}, illustration {image_prompt_index+1}"

            logger.info(message)
            await self._generate_system_request_to_update_user(message)

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
                    error_message = f"Failed to generate image for {'Cover' if chapter_number == 0 and image_prompt_index == 0 else f'Chapter {chapter_number}, Image {image_prompt_index}'}. Skipping."
                    logger.error(error_message)

                await self.create_and_send_progress_update(
                    story_state=str(StoryState.IMAGE_GENERATION),
                    status=ResponseStatus.STARTED,
                    progress_percent=60.0
                    + 0.3 * (image_generation_count / total_number_images) * 100,
                    # message="...",
                )

            except Exception as e:
                error_message = f"An error occurred while processing Chapter {chapter_number}, Image {image_prompt_index}."
                logger.error(f"{error_message}. Details:\n{traceback.format_exc()}")
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
        await self._generate_system_request_to_update_user(message)

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
            await self._generate_system_request_to_update_user(message)

            react_api_url = os.environ.get("REACT_APP_API_URL")
            url = f"{react_api_url}story/{self.story_manager.story.id}/download"
            logging.info(f"Story to download sent to frontend: {url}")
            await self.create_and_send_progress_update(
                command=Command.PROCESS_COMPLETED,
                story_state=str(StoryState.DOCUMENT_GENERATION),
                files=[url],
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
