import os
import re
from word2number import w2n
import copy
import asyncio
from typing import List, Tuple, Optional, Dict, Union, Any
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.utils import instantiate
import traceback
from sqlalchemy import desc, select, delete, update

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
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
from models.story import Story, StoryData
from models.command import Command
from models.response import ResponseStatus
from models.message import Message, MessageSchema, OriginEnum, TypeEnum
from models.profile import Profile
from models.user import User
from models.ws_input import WSInput
from models.ws_output import WSOutput

from db import transaction_context

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
        StoryState.STORY_TITLE_GENERATION: "_generate_story_title",
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
        self.chat_assistant = ChatAssistant(
            config=self.config.chat_assistant,
            command_handler=self.handle_assistant_requests,
        )

        # Subfolders for storing various assets
        self.subfolders = {"images": "images", "chapters": "chapters"}

        # Initialize story-related data with default or empty values
        self.story_data = StoryData()

        self.chapter_generation_mechanism = None

        self.latest_story_dir = None

        self.message_sender = message_sender
        self.session = session
        self.websocket = websocket
        self.token_data = None

        self.user_id = None
        self.user = None
        self.profile_id = None
        self.story_id = None
        self.new_token = None

        self.current_knowledge_base = {}
        
        logger.info("MagicTales initialized.")

    async def start(self):
        """
        Run the Magic Tales Core server.
        """
        logger.info("Running MagicTales.")
        #asyncio.run(self._run())

        await self.chat_assistant.start_assistant()
        await self.helper_assistant.start_assistant()


    def _validate_openai_api_key(self) -> None:
        """Validates the presence of the OpenAI API key."""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    async def process(self, request: WSInput, token_data: dict) -> None:
        """
        Process incoming commands from the AI Core Interface Layer ().

        Args:
            request (WSInput): Full command structure. Please look at the ICD docs for more details.
            token_data (dict): Token data from the AI Core Interface Layer.

        Returns:
            dict: Response data to be sent back to the AI Core Interface Layer.
        """
        logger.info(f"token:{request.token}")  # DEBUG
        
        if request.command == Command.NEW_TALE or request.command == Command.SPIN_OFF or request.command == Command.UPDATE_PROFILE:
            await self.send_message_to_frontend(
                    WSOutput(command=request.command, token=self.new_token, ack=True)
                )

        # Refresh access token and set user_id if isn't try_mode
        if request.try_mode is False or None:
            self.token_data = token_data
            self.new_token = await refresh_access_token(request.token)                        
            coming_user_id = token_data.get("user_id", None)            
        else:
            self.token_data = None
            self.new_token = None
            coming_user_id = None

        # There has been a change in the user_id        
        if self.user_id != coming_user_id:
            self.user_id = coming_user_id
            self.user = await self.get_user_by_id(self.user_id)
            
            user_message = self._generate_starting_message()
            await self._generate_and_send_update_message_for_user(user_message)

            if self.user:
                logger.info(f"User {self.user_id} has been set.")
                await self.chat_assistant.start_assistant(self.user)
                await self.helper_assistant.start_assistant(self.user)
                await self._fetch_user_data_and_create_knowledge_base()
            else:
                logger.warn(f"User {self.user_id} has not been found.")
                        

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
        await self._db_add_message_to_session(conversation)

        if request.command == Command.NEW_TALE:
            # await self.send_message_to_frontend(
            #     WSOutput(command=request.command, token=self.new_token, ack=True)
            # )
            asyncio.create_task(self._handle_new_tale(request))

        elif request.command == Command.SPIN_OFF:
            if not request.story_id:
                raise Exception("story_id is required for spin-off")

            # await self.send_message_to_frontend(
            #     WSOutput(
            #         command=request.command,
            #         token=self.new_token,
            #         ack=True,
            #         data={"story_parent_id": request.story_id},
            #     )
            # )
            asyncio.create_task(self._handle_spin_off_tale(request))

        elif request.command == Command.UPDATE_PROFILE:
            if not request.profile_id:
                raise Exception("profile_id is required for update profile")

            await self.send_message_to_frontend(
                WSOutput(command=request.command, token=self.new_token, ack=True)
            )
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
            conversations = await self.get_messages_by_session_id(self.websocket.uid)
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

            await self._db_link_user_with_conversations(request.session_ids)
            await self.send_message_to_frontend(
                WSOutput(command=request.command, token=self.new_token, ack=True)
            )

        elif request.command == Command.USER_MESSAGE:
            ai_response = await self._handle_user_message(request)
            # Send the response to the AI Core Interface Layer to be sent to the Front End
            await self.send_message_to_frontend(ai_response)

        logger.info(f"Processed command: {request.command}")

    
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

        # await self._db_add_message_to_session(message)
        

    # ---------------------------------------------- Getters methods ----------------------------------------------

    async def get_user_by_id(self, id) -> User:
        """
        Asynchronously retrieves a User by ID.
        """
        result = await self.session.get(User, id)
        return result

    async def get_profile_by_id(self, id) -> Profile:
        """
        Asynchronously retrieves a Profile by ID.
        """
        result = await self.session.get(Profile, id)
        return result

    async def get_story_by_id(self, id) -> Story:
        """
        Asynchronously retrieves a Story by ID.
        """
        result = await self.session.get(Story, id)
        return result

    async def get_stories_by_user_id(self, user_id) -> List[Story]:
        """
        Asynchronously retrieves Stories by a User's ID, through their Profiles.
        """
        profiles_result = await self.session.execute(
            select(Profile.id).where(Profile.user_id == user_id)
        )
        profiles_ids = profiles_result.scalars().all()

        stories_result = await self.session.execute(
            select(Story).filter(Story.profile_id.in_(profiles_ids))
        )
        return stories_result.scalars().all()

    async def get_stories_by_profile_id(self, profile_id) -> List[Story]:
        """
        Asynchronously retrieves Stories by Profile ID.
        """
        result = await self.session.execute(
            select(Story).where(Story.profile_id == profile_id)
        )
        return result.scalars().all()

    async def get_messages_by_session_id(self, session_id) -> List[Message]:
        """
        Asynchronously retrieves Messages by Session ID, filtering for 'chat' messages.
        """
        result = await self.session.execute(
            select(Message)
            .where(Message.session_id == session_id, Message.type == "chat")
            .order_by(desc(Message.session_id), Message.id)
        )
        return result.scalars().all()

    # ---------------------------------------------- End getters methods ------------------------------------------

    def create_progress_update(self, **kwargs) -> WSOutput:
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
            command=Command.STATUS_UPDATE
        )  # Adjust this based on your Command enum or definitions

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
        ai_response, messages = await self.chat_assistant.request_ai_response(
            message=user_message
        )
        logger.info(f"AI response: {ai_response}")

        # Construct a response to be sent by the AI Core Interface Layer to the Front End
        response = WSOutput(
            command=Command.MESSAGE_FOR_HUMAN,
            message=ai_response,  # AI-generated response to be displayed to the user
            token=self.new_token,
            working=False,  # Indicates that the response is ready
        )
        return response

    async def _handle_user_request_update_profile(self, request: WSInput) -> None:
        """
        Handle the 'user_request_update_profile' command by retrieving user information from the database, creating a chat where the user might night give us new information about this profile.

        Returns:
            None.
        """
        logger.info("Updating profile.")

        # self._reset()
        # await self._process_step(StoryState.USER_FACING_CHAT)

    async def _handle_spin_off_tale(self, request: WSInput) -> None:
        """
        Handle the 'spin_off' command by retrieving user information from the database.

        Returns:
            None.
        """
        logger.info("Starting a spin-off story generation.")        

        # self._reset()
        # await self._process_step(StoryState.USER_FACING_CHAT)

    async def _handle_new_tale(self, request: WSInput) -> None:
        """
        Handle the 'new-tale' command by retrieving user information from the database.

        Returns:
            None.
        """
        logger.info("Starting story generation from scratch.")        
        

        self._reset()
        await self._process_step(StoryState.USER_FACING_CHAT)

    async def _generate_and_send_update_message_for_user(self, request_message: str) -> None:
        """
        Sends an update message to the user.

        Args:
            request_message (str): This is the message we are sending the Assistant to generate an amazing update for the user.

        Returns:
            None.
        """
        # TODO: For some reason I cannot use self.user directly on the messages, I need to use the method to get the user
        user = await self.get_user_by_id(self.user_id)
        if not user:
            logger.error(f"User with id {self.user_id} not found.")
            user_info = "User not found"
        else:
            user_info = user.to_dict()

        request_message = request_message.replace("{user_info}", f"{user_info}")
        request_message = WSInput(
            command=Command.USER_MESSAGE, token=self.new_token, message=request_message
        )
        ai_response = await self._handle_user_message(request_message)
        
        await self.send_message_to_frontend(ai_response)
        

    async def _fetch_user_data_and_create_knowledge_base(self) -> None:
        """
        Retrieves user data from the database and creates a knowledge base file.

        Returns:
            List[str]: List of file paths to the created knowledge base files.
        """
        if self.user_id is None:
            return         

        await self._generate_and_send_update_message_for_user(self.config.updates_request_prompts.fetching_user_data)
                
        # Retrieve user data from the database
        user, profiles, stories = await self._query_user_data()
        self.current_knowledge_base = {
            "user_info": user,
            "profiles": profiles,
            "stories": stories,
        }

        files_paths = await convert_user_info_to_json_files(
            self.current_knowledge_base,
            self.config.output_artifacts.knowledge_base_root_dir,
        )
        # Ensure files_paths is a list, even if it's empty
        files_paths = files_paths or []
        await self.chat_assistant.update_assistant(files_paths=files_paths)

        await self._generate_and_send_update_message_for_user(self.config.updates_request_prompts.after_feching_user_data)
        

    async def _query_user_data(self) -> Tuple[User, List[Profile], List[Story]]:
        """
        Queries the database asynchronously for user information, profiles, and stories
        using SQLAlchemy.

        Returns:
            Tuple[User, List[Profile], List[Story]]: A tuple containing user information, profiles, and stories.
        """
        # Fetch user information
        user = self.user # await self.get_user_by_id(self.user_id)
        
        # Fetch profiles associated with the user
        profile_result = await self.session.execute(
            select(Profile).where(Profile.user_id == self.user_id)
        )
        profiles = profile_result.scalars().all()
        
        # Fetch stories associated with the user
        stories = await self.get_stories_by_user_id(self.user_id)        

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

    async def _resume_story(self) -> None:
        """
        Resumes the story generation process from the last saved state.
        """
        logger.info("Attempting to resume story generation from the last saved state.")
        latest_story_dir = get_latest_story_directory(
            self.config.output_artifacts.stories_root_dir
        )

        self.story_data = StoryData.load_state(directory=latest_story_dir)
        if self.story_data:
            last_step = self.story_data.metadata.get(
                "step", StoryState.USER_FACING_CHAT
            )
            if last_step is StoryState.FINAL_DOCUMENT_GENERATED:
                logger.info(f"Last story was completed. We will create a new story.")
                await self._start_new_story()
            else:
                logger.info(f"Resuming story generation from step: {last_step}")
            await self._process_step(last_step)
        else:
            logger.info("No saved state found, starting a new story.")
            await self._start_new_story()

    async def _process_step(self, current_step: StoryState, **kwargs) -> None:
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

                    # Move to the next step
                    current_step = StoryState.next(current_step)
                    self._update_and_serialize_step(current_step)
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

    def _reset(self) -> None:
        """
        Reset the variables for a fresh run.

        This method sets all internal state variables to their initial values.
        """
        try:
            # Resetting the StoryData object to its initial state
            self.story_data = StoryData()
            self.story_data.story_folder = create_new_story_directory(
                stories_root_dir=self.config.output_artifacts.stories_root_dir,
                subfolders=self.subfolders,
            )
            self.story_data.images_subfolder = self.subfolders["images"]
            # Log the reset action
            logger.info("State has been successfully reset for a fresh run.")

        except Exception as e:
            logger.error(
                f"An error occurred while resetting the state: {traceback.format_exc()}"
            )

    def _update_and_serialize_step(self, step: StoryState) -> None:
        self.story_data.metadata["step"] = step
        self.story_data.save_state()
        logger.info(f"Saved step: {step}")

    async def _user_facing_chat(self) -> None:
        """
        Start and run the WebSocket server to gather user information through a chat with the GUI.

        This method handles the user-facing chat phase, processes the completed chat,
        extracts story elements, and updates the database and knowledge base accordingly.
        """
        logger.info("Starting User-Facing chat Phase")        

        # Wait for chat to complete
        await self.chat_assistant.wait_for_chat_completion()

        # Process the completed chat
        await self.on_chat_completed(await self.chat_assistant._retrieve_messages())

    def _generate_starting_message(self) -> str:
        """
        Generates a starting message for the user based on the presence of a knowledge base.

        Returns:
            str: The starting message for the user.
        """        
        if self.user:
            return self.config.updates_request_prompts.starting_message_existing_user
        else:
            return self.config.updates_request_prompts.starting_message_new_user

    async def handle_assistant_requests(self, ai_message_for_system: WSInput) -> None:
        """
        Processes the Chat AI Assistant message.
        """
        logger.info(
            f"Processing AI message for Magic-Tales Orchestrator: {ai_message_for_system}"
        )
        if ai_message_for_system.command == Command.CHAT_COMPLETED:
            self.profile_id = ai_message_for_system.profile_id
            self.chat_assistant.chat_completed_event.set()
            logger.info("Chat completed.")
            await self._generate_and_send_update_message_for_user(self.config.updates_request_prompts.chat_completed)
            return

        if not isinstance(ai_message_for_system, WSInput):
            logger.info(
                f"AI message for Magic-Tales Orchestrator is not a WSInput: {ai_message_for_system}"
            )
            return

        command = ai_message_for_system.command

        if command == Command.UPDATE_PROFILE:
            await self._generate_and_send_update_message_for_user(self.config.updates_request_prompts.updating_profile)
            await self._db_update_profile_from_chat(ai_message_for_system)
        elif command == Command.NEW_PROFILE:
            await self._generate_and_send_update_message_for_user(self.config.updates_request_prompts.creating_new_profile)            
            await self._db_create_profile_from_chat(ai_message_for_system)

    # ---------------------------------------------- DB Post methods -------------------------------------------------

    async def _db_create_profile_from_chat(self, ai_message_for_system: WSInput):

        if not self.user_id:
            raise Exception("user_id is required for create profile")

        profile_details = ai_message_for_system.message

        new_profile = Profile(
            user_id=self.user_id,
            user=self.user, # await self.get_user_by_id(self.user_id),
            details=profile_details,
        )
        self._db_create_profile(new_profile)

        return new_profile

    async def _db_create_profile(self, new_profile: Profile) -> Profile:
        """
        Asynchronously creates a new profile in the database.

        Args:
            new_profile (Profile): The profile instance to be added to the database.

        Returns:
            Profile: The newly created profile.

        Raises:
            Exception: If `user_id` or `new_profile` is not set.
        """
        if not self.user_id:
            raise Exception("user_id is required to create a profile")
        if not new_profile:
            raise Exception("new_profile is required for create profile")

        try:
            logger.debug("Starting transaction...")
            async with transaction_context(self.session):
                self.session.add(new_profile)
            logger.debug("Transaction committed.")
            return new_profile
        except SQLAlchemyError as e:
            # Log the exception here
            raise Exception(
                f"Failed to create profile: {e}/n/n{traceback.format_exc()}"
            )

    async def _db_create_story(self, data: Story) -> Story:
        """
        Asynchronously creates a new story in the database.

        Args:
            data (Story): The story data to be added.

        Returns:
            Story: The newly created story with updated ID.
        """
        try:
            logger.debug("Starting transaction...")
            async with transaction_context(self.session):
                self.session.add(data)
            await self.session.refresh(data)
            logger.debug("Transaction committed.")
            return data
        except SQLAlchemyError as e:
            # Log the exception here
            raise Exception(f"Failed to create story: {e}/n/n{traceback.format_exc()}")

    async def _db_add_message_to_session(self, data: Message) -> Message:
        """
        Asynchronously adds a new message to a session in the database.

        Args:
            data (Message): The message data to be added.

        Returns:
            Message: The newly added message with updated ID.
        """
        try:
            logger.debug("Starting transaction...")
            async with transaction_context(self.session):
                self.session.add(data)
            logger.debug("Transaction committed.")

            await self.session.refresh(data)

            return data
        except SQLAlchemyError as e:
            # Log the exception here
            raise Exception(
                f"Failed to add message to session: {e}/n/n{traceback.format_exc()}"
            )

    async def _db_delete_messages_by_session_id(self, session_id: int):
        """
        Asynchronously deletes messages by session ID. (Not used)

        Args:
            session_id (int): The ID of the session for which messages should be deleted.
        """
        try:
            logger.debug("Starting transaction...")
            async with transaction_context(self.session):
                await self.session.execute(
                    delete(Message).where(Message.session_id == session_id)
                )
            logger.debug("Transaction committed.")
        except SQLAlchemyError as e:
            # Log the exception here
            raise Exception(
                f"Failed to delete messages: {e}/n/n{traceback.format_exc()}"
            )

    
    async def _db_update_user_by_id(self, user_id: str, data: dict) -> User:
        """
        Asynchronously updates a user by ID.

        Args:
            user_id (str): The ID of the user to update.
            data (dict): A dictionary with user fields to update.

        Returns:
            User: The updated user.
        """
        try:
            logger.debug("Starting transaction...")
            async with transaction_context(self.session):
                user = await self.session.get(User, user_id)
                for key, value in data.items():
                    setattr(user, key, value)            
            logger.debug("Transaction committed.")
            return user
        except SQLAlchemyError as e:
            # Log the exception here
            raise Exception(
                f"Failed to update user: {e}/n/n{traceback.format_exc()}"
            )

    async def _db_update_profile_by_id(self, profile_id: str, data: dict) -> Profile:
        """
        Asynchronously updates a profile by ID.

        Args:
            profile_id (str): The ID of the profile to update.
            data (dict): A dictionary with profile fields to update.

        Returns:
            Profile: The updated profile.
        """
        try:
            logger.debug("Starting transaction...")
            async with transaction_context(self.session):
                profile = await self.session.get(Profile, profile_id)
                for key, value in data.items():
                    setattr(profile, key, value)            
            logger.debug("Transaction committed.")
            return profile
        except SQLAlchemyError as e:
            # Log the exception here
            raise Exception(
                f"Failed to update profile: {e}/n/n{traceback.format_exc()}"
            )

    async def _db_update_story_by_id(self, story_id: str, data: dict) -> Story:
        """
        Asynchronously updates a Story by ID.

        Args:
            story_id (str): The ID of the profile to update.
            data (dict): A dictionary with story fields to update.

        Returns:
            Profile: The updated story.
        """
        try:
            logger.debug("Starting transaction...")
            async with transaction_context(self.session):
                story = await self.session.get(Story, story_id)
                for key, value in data.items():
                    setattr(story, key, value)            
            logger.debug("Transaction committed.")
            return story
        except SQLAlchemyError as e:
            # Log the exception here
            raise Exception(
                f"Failed to update story: {e}/n/n{traceback.format_exc()}"
            )
        
    async def _db_link_user_with_conversations(self, session_ids: list):
        """
        Asynchronously links a user with conversations by updating messages with the user ID.

        Args:
            session_ids (list): A list of session IDs to update messages with the user ID.
        """
        try:
            logger.debug("Starting transaction...")
            async with transaction_context(self.session):
                await self.session.execute(
                    update(Message)
                    .where(Message.session_id.in_(session_ids))
                    .values(user_id=self.user_id)
                )
            logger.debug("Transaction committed.")
        except SQLAlchemyError as e:
            # Log the exception here
            raise Exception(
                f"Failed to link user with conversations: {e}/n/n{traceback.format_exc()}"
            )

    # ---------------------------------------------- End update methods -------------------------------------------

    async def _db_update_profile_from_chat(self, ai_message_for_system: WSInput):
        """
        Updates a profile based on the AI system message.

        Args:
            ai_message_for_system (WSInput): The message from the AI system with update details.
        """
        profile_id = ai_message_for_system.profile_id
        updates = ai_message_for_system.message

        # Fetch existing profile details
        current_profile = await self.get_profile_by_id(profile_id)
        existing_details = current_profile.details if current_profile else ""

        # Merge updates using the Information Extractor Assistant
        merged_details = await self.merge_profile_updates(existing_details, updates)

        # Update the profile in the database
        updated_profile = Profile(
            user_id=self.user_id,
            user=self.user, # await self.get_user_by_id(self.user_id),
            details=merged_details,
        )
        await self._db_update_profile_by_id(profile_id, updated_profile)

        # Update the knowledge base files
        self.current_knowledge_base["profiles"] = merged_details        

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
        merged_details , _ = await self.helper_assistant.request_ai_response(chat_string)
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

        await self._generate_and_send_update_message_for_user(self.config.updates_request_prompts.chat_info_extraction)            

        chat_string = await self._convert_chat_to_string(messages)
        updated_elements = await self._extract_chat_key_elements(chat_string)

        # TODO: If we are in Try_mode we need to go to registration and then get back here to continue the process
        # if self.user_id:
        logger.info("Updating the database with new story elements from chat")
        await self._create_story_foundation_from_chat_key_elements(updated_elements)            
        # await self._db_update_user_by_id(self.user_id, {"assistant_id": self.chat_assistant.openai_assistant.id})
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
        if self.user_id is None:
            personality_profile = await self.helper_assistant.request_ai_response(
            f"Extract a detailed list of all related features of the individual we want to write the story for from this conversation:\n{chat_string}")
        else:
            personality_profile = None

        story_features = await self.helper_assistant.request_ai_response(
                f"Identify story features from this conversation:\n{chat_string}"
            )
        
        story_synopsis = await self.helper_assistant.request_ai_response(
                f"Extract the latest agreed upon story synopsis from this conversation:\n{chat_string}"
            )
        tasks = {
            "personality_profile": personality_profile,
            "story_features": story_features,
            "story_synopsis": story_synopsis,
        }

        return tasks
    
    async def _create_story_foundation_from_chat_key_elements(self, updates: Dict):
        """
        Updates the database with the provided updates and refreshes the knowledge base.

        Parameters:
        - updates: Dict[str, str, str] - The updates to apply, keyed by the type of update
          ('personality_profile', 'story_features', 'story_synopsis') and containing the new values.
        """
        # if "personality_profile" in updates:
        #     profile_details = updates["personality_profile"]
        #     new_profile = Profile(
        #         id=self.profile_id,
        #         user_id=self.user_id,
        #         user=await self.get_user_by_id(self.user_id),
        #         details=profile_details,
        #     )
        #     await self._db_update_profile_by_id(new_profile)

        personality_profile = updates.get("personality_profile", None)
        if not personality_profile:
            current_profile = await self.get_profile_by_id(self.profile_id)
            personality_profile = current_profile.details if current_profile else ""
        self.story_data.personality_profile = personality_profile

        story_features = updates.get("story_features", "")
        self.story_data.story_features = story_features

        story_synopsis = updates.get("story_synopsis", "")
        self.story_data.synopsis = story_synopsis

        # if "story_features" in updates or "story_synopsis" in updates:
        #     new_story = Story(
        #         profile_id=self.profile_id,
        #         features=story_features,
        #         synopsis=story_synopsis,
        #         last_successful_step=StoryState.USER_FACING_CHAT.value,
        #     )
        #     story = await self._db_create_story(new_story)
        #     self.story_id = story.id

        logger.info("Story foundation created on the Database with new story elements from chat.")

    async def _generate_story_title(self) -> None:
        """Generate a title for the story based on the user input."""
        logger.info(f"Generating a title for the story")
        await self._generate_and_send_update_message_for_user(self.config.updates_request_prompts.generating_story_title)

        message = f"Given the following three pieces of information:\n\n1) Personality profile: {self.story_data.personality_profile}.\n\n2) Story main features: {self.story_data.story_features}.\n\n3) Story Synopsis: {self.story_data.synopsis}.\n\nThe best possible title for the story is (just respond with the Title, nothing else):"
        self.story_data.title, _ = await self.helper_assistant.request_ai_response(message)
        logger.info(f"Title: {self.story_data.title}")

        # Update the story database record with the generated title
        # story = await self._db_update_story_by_id(self.story_id, {"title": self.story_data.title})
        # self.story_id = story.id

        update = self.create_progress_update(
            story_state=StoryState.STORY_TITLE_GENERATION,
            status=ResponseStatus.STARTED,
            progress_percent=100,
            message="We have generated a title for the story.",
        )
        await self.send_message_to_frontend(update)

        return

    async def _generate_chapter_title(self, chapter: str) -> str:
        """Generate a title for specific Chapter"""
        logger.info(f"Generating a title for the Chapter")
        message = f"Given the following piece of information:\n\n1) Chapter: {chapter}.\n\nThe best possible title for this chapter is (just respond with the title, nothing else):"
        chapter_title , _ = await self.helper_assistant.request_ai_response(message)
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

        await self._generate_and_send_update_message_for_user(self.config.updates_request_prompts.generating_story)            

        # Initialize a list to hold the chapters
        chapters = []

        # Step 1: Determine Chapter Count
        # TODO: Make this chapter count more robust
        num_chapters = await self._determine_chapter_count(
            self.story_data.story_features
        )
        num_chapters = max(1, num_chapters)
        logger.info(f"Number of chapters: {num_chapters}")

        # Step 2: Initialize chapter content
        previous_chapter_content = ""

        # Step 3: Chapter Generation
        chapters_folder = os.path.join(
            self.story_data.story_folder, self.subfolders["chapters"]
        )
        for chapter_number in range(1, num_chapters + 1):
            message = f"Generating chapter: {chapter_number} of {num_chapters}"
            logger.info(message)

            update = self.create_progress_update(
                story_state=StoryState.STORY_GENERATION,
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

        self.story_data.chapters = chapters

        # Update the story database record with the chapters
        # story = await self._db_update_story_by_id(self.story_id, {"chapters": chapters})
        # self.story_id = story.id

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
            story_data=self.story_data,
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
            chapter_title = self.story_data.title

        logger.info(f"Chapter {chapter_number}: {chapter_title}")
        logger.info(f"{chapter_content}")

        return chapter_title, chapter_content

    async def _generate_image_prompts(self) -> None:
        """Generate image prompts for all chapters in the story."""
        message = f"Generating image prompts for all chapters."
        logger.info(message)
        update = self.create_progress_update(
            story_state=StoryState.IMAGE_PROMPT_GENERATION,
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
                chapters=self.story_data.chapters
            )
        )
        await self._extract_post_processed_chapters(
            all_chapter_prompts_and_new_chapters_annotated
        )
        logger.info(
            f"Image Prompts + processed chapters created"
        )  # {self.story_data.image_prompts}")

    async def _extract_post_processed_chapters(
        self, all_chapter_prompts: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Save the post-processed chapters, which include image annotations, into the StoryData structure.

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

        # Save the post-processed chapters to the StoryData structure.
        self.story_data.post_processed_chapters = post_processed_chapters
        self.story_data.image_prompt_generator_prompt_messages = image_prompt_messages
        self.story_data.image_prompts = image_prompts

    async def _generate_images(self) -> List[str]:
        """
        Generate images based on provided image prompts for all chapters and save them to a specified directory.

        Returns:
            A list of filenames for the generated images.

        Raises:
            Exception: If any error occurs during the image generation process.
        """
        logger.info("Initiating image generation process.")

        all_chapter_prompts = self.story_data.image_prompts
        story_directory = self.story_data.story_folder

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
            update = self.create_progress_update(
                story_state=StoryState.IMAGE_GENERATION,
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
                    update = self.create_progress_update(
                        story_state=StoryState.IMAGE_GENERATION,
                        status=ResponseStatus.STARTED,
                        progress_percent=0,
                        message=message,
                    )
                    await self.send_message_to_frontend(update)
            except Exception as e:
                message = f"An error occurred while processing Chapter {chapter_number}, Image {image_prompt_index}."
                logger.error(f"{message}. Details:\n{traceback.format_exc()}")
                update = self.create_progress_update(
                    story_state=StoryState.IMAGE_GENERATION,
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

    async def _generate_final_document(self) -> None:
        """Creates the final story document with associated images and saves it to the specified directory."""
        message = f"Generating the final story document."
        logger.info(message)
        update = self.create_progress_update(
            story_state=StoryState.DOCUMENT_GENERATION,
            status=ResponseStatus.STARTED,
            progress_percent=0,
            message=message,
        )
        await self.send_message_to_frontend(update)

        story_document = StoryDocument(story_data=self.story_data)
        story_document.create_document()
        filepath = story_document.save_document(f"story.docx")
        logger.info(f"Story document saved in: {filepath}")

        message = f"Story generation completed. Hope you enjoy it!"
        logger.info(message)
        update = self.create_progress_update(
            story_state=StoryState.DOCUMENT_GENERATION,
            status=ResponseStatus.STARTED,
            progress_percent=0,
            message=message,
        )
        await self.send_message_to_frontend(update)
