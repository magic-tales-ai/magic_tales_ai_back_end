from typing import Any
import traceback
import os
import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Callable
from openai import AsyncOpenAI
from openai.types.beta.threads import (
    ThreadMessage,
    MessageContentText,
    MessageContentImageFile,
)

from services.utils.log_utils import get_logger
from services.chat_assistant.prompt_utils import (
    async_load_prompt_template_from_file,
)
from models.ws_input import WSInput
from models.user import User

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class ChatAssistant:
    def __init__(self, config, command_handler: Callable):
        """
        Initialize the Chat Assistant.

        Args:
            config (DictConfig): Configuration parameters.
            command_handler (function): Function to handle messages for the Orchestrator sent by the AI Assistant: DB commands mainly.
        """
        self._validate_openai_api_key()
        self.config = config
        self.client = AsyncOpenAI()
        self.openai_assistant = None
        self.user_facing_thread = None
        self.user_facing_chat_info = []
        self.chat_completed_event = asyncio.Event()
        self.orchestrsator_command_handler = command_handler

    def _validate_openai_api_key(self):
        """Validates the presence of the OpenAI API key."""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    async def _create_assistant(self, user_data: User = None):
        """
        Creates a NEW OpenAI assistant with attached knowledge base files.

        """
        template_prompt = await async_load_prompt_template_from_file(
            self.config.instructions_tryout_path
        )
        tryout_instructions = template_prompt.replace("<user_data>", str(user_data))
        self.openai_assistant = await self.client.beta.assistants.create(
            name="Smarty Tales",
            instructions=tryout_instructions,
            model=self.config.model,
            tools=[{"type": "retrieval"}],
            file_ids=self.file_ids or [],        
        )
        self.user_facing_thread = await self.client.beta.threads.create()
        logger.info("OpenAI Assistant Created.")

    async def _retrieve_assistant(self, user_data: User = None):
        """
        Retrieves an existing OpenAI assistant with attached knowledge base files.

        """
        if not user_data or not user_data.assistant_id:
            logger.warning("Assistant ID is required for retrieval.")
            await self._create_assistant(user_data)
            return

        template_prompt = await async_load_prompt_template_from_file(
            self.config.instructions_tryout_path
        )
        instructions = template_prompt.replace("<user_data>", str(user_data))

        self.openai_assistant = await self.client.beta.assistants.update(
            assistant_id=user_data.assistant_id,
            instructions=instructions,
            model=self.config.model,
            tools=[{"type": "retrieval"}],
            file_ids=self.file_ids or [],
        )

        self.user_facing_thread = await self.client.beta.threads.create()
        logger.info("OpenAI Assistant Retrieved.")

    async def _update_assistant(self):
        """
        Initializes the OpenAI assistant with attached knowledge base files.

        """
        if not id:
            raise ValueError("Assistant ID is required for updating.")

        self.openai_assistant = await self.client.beta.assistants.update(
            assistant_id=self.openai_assistant.id,
            file_ids=self.file_ids or [],
        )
        logger.info("OpenAI Assistant UPDATED.")

    async def wait_for_chat_completion(self):
        """
        Waits for the chat to complete. This method blocks until the chat_complete_event is set.
        """
        await self.chat_completed_event.wait()

    async def generate_ai_response(
        self, message: str, parsing_method: Optional[Callable] = None
    ) -> Tuple[str, List]:
        """
        Process the incoming message and generate an AI response.

        The function sends the message to the OpenAI assistant and retrieves the response.
        It handles the entire process including sending the message, initiating the run, waiting
        for completion, and processing the received response.

        Args:
            message (str): The message received from the client.
            parsing_method (Callable, optional): A method to parse the AI response.

        Returns:
            Tuple[str, List]: A tuple containing the response message for the human and
                              the raw data of messages.

        Raises:
            Exception: If the OpenAI API call fails or the parsing method encounters an error.
        """
        messages_data = []

        try:
            # Send the message to the OpenAI assistant
            await self._send_message_to_assistant(message)
            # Initiate the assistant run and wait for completion
            run = await self._wait_for_assistant_run_completion()

            # Retrieve the latest messages from the assistant
            messages_data = await self._retrieve_messages()

            # Extract the latest AI response message
            ai_message_for_human, ai_message_for_system = (
                await self._process_ai_response(messages_data, parsing_method)
            )

            # Handle any commands for the system included in the AI response
            if ai_message_for_system:
                await self.orchestrsator_command_handler(ai_message_for_system)

            return ai_message_for_human, messages_data

        except Exception as e:
            logger.error(
                f"An error occurred during AI response generation: {e}/n/n{traceback.format_exc()}", exc_info=True
            )
            return "", messages_data

    async def _send_message_to_assistant(self, message: str) -> None:
        """
        Sends the user's message to the OpenAI assistant thread.

        Args:
            message (str): The message to send to the assistant.

        Raises:
            Exception: If sending the message fails.
        """
        await self.client.beta.threads.messages.create(
            thread_id=self.user_facing_thread.id,
            role="user",
            content=message,
        )
        logger.info(f"Message sent to OpenAI Assistant: {message}")

    async def _wait_for_assistant_run_completion(self) -> Any:
        """
        Initiates a run with the OpenAI assistant and waits for it to complete.

        Returns:
            Any: The run object after completion.

        Raises:
            Exception: If initiating or waiting for the run fails.
        """
        run = await self.client.beta.threads.runs.create(
            thread_id=self.user_facing_thread.id,
            assistant_id=self.openai_assistant.id,
        )
        logger.info("Assistant run initiated, waiting for completion...")
        while run.status in ["queued", "in_progress"]:
            await asyncio.sleep(0.5)  # Non-blocking sleep
            run = await self.client.beta.threads.runs.retrieve(
                thread_id=self.user_facing_thread.id,
                run_id=run.id,
            )
        logger.info("Assistant run completed.")
        return run

    async def _retrieve_messages(self) -> List:
        """
        Retrieves the list of messages from the OpenAI assistant thread in descending order.

        Returns:
            List: The list of messages data.

        Raises:
            Exception: If retrieving the messages fails.
        """
        messages = await self.client.beta.threads.messages.list(
            thread_id=self.user_facing_thread.id, order="desc"
        )
        logger.info("Retrieved messages from OpenAI Assistant.")
        return messages.data

    async def _process_ai_response(
        self, messages_data: List, parsing_method: Optional[Callable] = None
    ) -> Tuple[str, Any]:
        """
        Processes the OpenAI assistant's latest response.

        Args:
            messages_data (List): The list of messages data from the assistant.
            parsing_method (Callable, optional): A method to parse the AI response.

        Returns:
            Tuple[str, Any]: A tuple of the AI's response for the human and the system command.

        Raises:
            Exception: If processing the response fails.
        """
        # Process the response to find the latest AI message
        ai_message = next((m for m in messages_data if m.role == "assistant"), None)
        if not ai_message:
            raise Exception("No AI message found.")

        ai_message_content = ai_message.content[0].text.value
        if parsing_method:
            return parsing_method(ai_message_content)
        else:
            return self._default_parsing(ai_message_content)

    # def _default_parsing(self, ai_message_content: str) -> Tuple[str, Any]:
    #     """
    #     Default parsing method for the AI response content.

    #     Args:
    #         ai_message_content (str): The raw AI response content.

    #     Returns:
    #         Tuple[str, Any]: A tuple of the AI's response for the human and the system command.

    #     Raises:
    #         json.JSONDecodeError: If parsing the JSON content fails.
    #     """
    #     try:
    #         # Attempt to parse the JSON response
    #         ai_message_dict = json.loads(ai_message_content)
    #         ai_message_for_human = ai_message_dict.get("message_for_human", "")

    #         ai_message_for_system = None

    #         if ai_message_dict.get("message_for_system"):
    #             ai_message_for_system = WSInput(**ai_message_dict.get("message_for_system"))

    #         return ai_message_for_human, ai_message_for_system
    #     except json.JSONDecodeError:
    #         # Fallback to regex if JSON parsing fails
    #         logger.info(f"Error processing AI message. Trying fallback method (re): {ai_message_content}")
    #         try:
    #             message_for_human = re.search(r'"message_for_human":\s*"([^"]*)"', ai_message_content).group(1)
    #             message_for_system = re.search(r'"message_for_system":\s*"([^"]*)"', ai_message_content).group(1)
    #             return message_for_human, message_for_system
    #         except Exception as e:
    #             logger.error(f"Error processing AI message: {traceback.format_exc()}")
    #             return ai_message_content, None

    def _extract_with_fallbacks(
        self, ai_message_content: str
    ) -> Tuple[Optional[str], Optional[dict]]:
        """
        Attempts to directly parse AI message content as JSON and extract `message_for_human` and `message_for_system`.

        Args:
            ai_message_content (str): The raw AI response content.

        Returns:
            A tuple containing potentially extracted `message_for_human` as str and
            `message_for_system` as a dict or None for each if not applicable or errors occur.
        """
        try:
            # Directly parse the AI message content as JSON
            ai_message_dict = json.loads(ai_message_content)
            message_for_human = ai_message_dict.get("message_for_human")
            message_for_system = ai_message_dict.get(
                "message_for_system"
            )  # This will be a dict or None

            return message_for_human, message_for_system

        except json.JSONDecodeError as e:
            try:
                message_for_human = re.search(
                    r'"message_for_human":\s*"([^"]*)"', ai_message_content
                ).group(1)
                message_for_system = re.search(
                    r'"message_for_system":\s*"([^"]*)"', ai_message_content
                ).group(1)
                return message_for_human, message_for_system
            except Exception as e:
                logger.error(
                    f"Failed to parse AI message content: {ai_message_content}/n/nError traceback:/n/n {traceback.format_exc()}"
                )
                retry_message = " I apologize, I encountered a hiccup processing your message. Could you rephrase or try again?"
                message_for_human = (message_for_human or "") + retry_message
                return message_for_human, None

    def _default_parsing(
        self, ai_message_content: str
    ) -> Tuple[Optional[str], Optional[WSInput]]:
        """
        Robustly parses AI response content, extracting `message_for_human` and creating a WSInput instance for `message_for_system`.
        """
        message_for_human, message_for_system_pre_processed = (
            self._extract_with_fallbacks(ai_message_content)
        )

        message_for_system = None
        if message_for_system_pre_processed:
            try:
                # Check if message_for_system_str is a string and attempt to parse it into a dict
                if isinstance(message_for_system_pre_processed, str):
                    message_for_system_dict = json.loads(
                        message_for_system_pre_processed
                    )
                elif isinstance(message_for_system_pre_processed, dict):
                    message_for_system_dict = message_for_system_pre_processed
                else:
                    raise ValueError(
                        "message_for_system is neither a dict nor a stringifiable JSON."
                    )

                # Now create WSInput instance from the properly parsed dict
                message_for_system = WSInput(**message_for_system_dict)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"Error creating WSInput from system message: {e}")
                # User-friendly message in case of failure
                retry_message = " I apologize, I encountered a hiccup processing a system request. Could you rephrase or try again?"
                message_for_human = (message_for_human or "") + retry_message

        return message_for_human, message_for_system

    async def create_and_upload_knowledge_files(
        self, files_paths: List[str]
    ) -> List[str]:
        """
        Uploads knowledge base files to OpenAI and returns their IDs.

        Args:
            files_paths (List[str]): Paths to the knowledge base files.

        Returns:
            List[str]: A list of file IDs after upload.
        """
        # Ensure files_paths is a list, even if it's empty
        files_paths = files_paths or []
        file_ids = []
        for file_path in files_paths:
            with open(file_path, "rb") as file:
                uploaded_file = await self.client.files.create(
                    file=file, purpose="assistants"
                )
                file_ids.append(uploaded_file.id)
        return file_ids

    async def update_and_upload_knowledge_files(
        self, files_paths: List[str]
    ) -> List[str]:
        """
        UPDATES the changed files and Uploads knowledge base files to OpenAI and returns their IDs.

        Args:
            files_paths (List[str]): Paths to the new knowledge base files.

        Returns:
            List[str]: A list of file IDs after upload.
        """
        # First delete the old files
        for file_id in self.file_ids:
            await self.client.files.delete(file_id)
        self.file_ids = []

        # Then upload the new files
        return await self.create_and_upload_knowledge_files(files_paths)

    async def start_assistant(
        self, user_data: User = None, files_paths: List[str] = []
    ):
        """
        Starts the chat assistant.
        """
        try:
            self.file_ids = await self.create_and_upload_knowledge_files(files_paths)
            if user_data:
                await self._retrieve_assistant(user_data)
            else:
                await self._create_assistant(user_data)

            logger.info("Chat Assistant initialized.")

        except Exception as e:
            logger.error(f"Error initializing assistant: {traceback.format_exc()}")
            raise e

    async def update_assistant(self, files_paths: List[str]):
        """
        Starts the chat assistant.
        """
        try:
            self.files_paths = files_paths
            self.file_ids = await self.update_and_upload_knowledge_files(
                self.files_paths
            )
            await self._update_assistant()

        except Exception as e:
            logger.error(f"Error initializing assistant: {traceback.format_exc()}")
            raise e
