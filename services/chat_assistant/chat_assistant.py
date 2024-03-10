import traceback
import os
import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Callable
from openai import AsyncOpenAI
from services.utils.log_utils import get_logger
from services.chat_assistant.prompt_utils import (
    async_load_prompt_template_from_file,
)

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
        self.user_facing_assistant = None
        self.user_facing_thread = None
        self.user_facing_chat_info = []
        self.chat_completed_event = asyncio.Event()
        self.orchestrsator_command_handler = command_handler

    def _validate_openai_api_key(self):
        """Validates the presence of the OpenAI API key."""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    async def _create_assistant(self):
        """
        Initializes the OpenAI assistant with attached knowledge base files.

        """
        template_prompt = await async_load_prompt_template_from_file(
            self.config.instructions_path
        )
        instructions = template_prompt.replace(
            "<user_data>", "It's all included in the attached files."
        )
        self.user_facing_assistant = await self.client.beta.assistants.create(
            name="Smarty Tales",
            instructions=instructions,
            model=self.config.model,
            tools=[{"type": "retrieval"}],
            file_ids=self.file_ids or [],
        )
        self.user_facing_thread = await self.client.beta.threads.create()
        logger.info("OpenAI Assistant initialized.")

    async def _update_assistant(self):
        """
        Initializes the OpenAI assistant with attached knowledge base files.

        """
        self.user_facing_assistant = await self.client.beta.assistants.update(
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

        This method is a placeholder for integrating the OpenAI assistant logic.

        Args:
            message (str): The message received from the client.
            parsing_method (Callable): A method to parse the AI response.

        Raises:
            Exception: If the OpenAI API call fails or no valid response is received.

        Returns:
            str: The response message to be sent back to the client.
        """
        try:
            # Create the message with the user's input (message)
            await self.client.beta.threads.messages.create(
                thread_id=self.user_facing_thread.id,
                role="user",
                content=message,
            )

            # Initiate the run
            run = await self.client.beta.threads.runs.create(
                thread_id=self.user_facing_thread.id,
                assistant_id=self.user_facing_assistant.id,
            )

            # Wait for the run to complete
            while run.status in ["queued", "in_progress"]:
                await asyncio.sleep(0.5)  # Non-blocking sleep
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=self.user_facing_thread.id,
                    run_id=run.id,
                )

            # Retrieve the latest assistant message after the run has completed
            messages = await self.client.beta.threads.messages.list(
                thread_id=self.user_facing_thread.id, order="desc"
            )

            # Process the response to find the latest AI message
            ai_message = await self._find_latest_ai_response(messages.data)
            try:
                ai_message_content = (
                    ai_message.content[0].text.value if ai_message else ""
                )
                ai_message_dict = json.loads(ai_message_content)
                ai_message_for_human = ai_message_dict.get("message_for_human", "")
                ai_message_for_system = ai_message_dict.get("message_for_system", {})
                await self.orchestrsator_command_handler(ai_message_for_system)

                return ai_message_for_human, messages.data

            except Exception as e:
                logger.error(f"Error processing AI message: {traceback.format_exc()}")
                return ai_message_content, messages.data

        except Exception as e:
            logger.error(f"Error processing AI message: {traceback.format_exc()}")
            return "", messages.data

    async def _find_latest_ai_response(self, messages: List) -> Optional[dict]:
        """
        Find the latest AI response from a list of messages.

        Args:
            messages (List): A list of message dicts in reverse-chronological order.

        Returns:
            Optional[dict]: The latest message dict from the AI or None if not found.
        """
        for message in messages:
            if message.role == "assistant":
                return message
        return None

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

    async def start_assistant(self, files_paths: List[str]):
        """
        Starts the chat assistant.
        """
        try:
            self.file_ids = await self.create_and_upload_knowledge_files(files_paths)
            await self._create_assistant()
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
