from abc import ABC, abstractmethod
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
from services.openai_assistants.prompt_utils import (
    async_load_prompt_template_from_file,
)
from models.ws_input import WSInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class Assistant(ABC):
    def __init__(self, config):
        """
        Initialize the OpenAI Assistant.

        Args:
            config (DictConfig): Configuration parameters.
        """
        self._validate_openai_api_key()
        self.config = config
        self.client = AsyncOpenAI()
        self.openai_assistant = None
        self.openai_thread = None
        self.file_ids = []
        self.retry_count = 0
        logger.info(f"{self.config.name} AI assistant class initialized.")

    def _validate_openai_api_key(self):
        """Validates the presence of the OpenAI API key."""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    async def _create_assistant(self):
        """
        Creates a NEW OpenAI assistant with attached knowledge base files.

        """
        if not self.config.instructions_path:
            raise ValueError("Instructions path is required for assistant creation.")

        instructions = await async_load_prompt_template_from_file(
            self.config.instructions_path
        )

        self.openai_assistant = await self.client.beta.assistants.create(
            name=self.config.name,
            instructions=instructions,
            model=self.config.model,
            tools=self.config.tools,
            file_ids=self.file_ids or [],
        )
        self.openai_thread = await self.client.beta.threads.create()
        logger.info(f"{self.config.name} OpenAI Assistant Created.")

    async def _retrieve_assistant(self, assistant_id=None):
        """
        Retrieves an existing OpenAI assistant with attached knowledge base files.

        """
        if not assistant_id:
            logger.warning(
                "Assistant ID is required for retrieval!! We will create a new Assistant."
            )
            await self._create_assistant()
            return

        try:
            if not self.config.instructions_path:
                raise ValueError(
                    "Instructions path is required for assistant creation."
                )

            instructions = await async_load_prompt_template_from_file(
                self.config.instructions_path
            )
            self.openai_assistant = await self.client.beta.assistants.update(
                assistant_id=assistant_id,
                instructions=instructions,
                model=self.config.model,
                tools=self.config.tools,
                file_ids=self.file_ids or [],
            )
        except:
            logger.error(
                f"We failed to retrive assistant ID: {assistant_id} from openAI. We will create a new Assistant."
            )
            await self._create_assistant()
            return

        self.openai_thread = await self.client.beta.threads.create()
        logger.info(f"{self.config.name} OpenAI Assistant retrieved succesfuly.")

    async def _update_assistant(self):
        """
        Initializes the OpenAI assistant with attached knowledge base files.

        """
        if not self.openai_assistant or not self.openai_assistant.id:
            raise ValueError("Assistant ID is required for updating.")

        self.openai_assistant = await self.client.beta.assistants.update(
            assistant_id=self.openai_assistant.id,
            file_ids=self.file_ids or [],
        )
        logger.info(f"{self.config.name} OpenAI Assistant UPDATED.")

    async def request_ai_response(
        self, message: str, parsing_method: Optional[Callable] = None
    ) -> Tuple[str, str]:
        """
        Process the incoming message and generate an AI response.

        The function sends the message to the OpenAI assistant and retrieves the response.
        It handles the entire process including sending the message, initiating the run, waiting
        for completion, and processing the received response.

        Args:
            message (str): The message received from the client.
            parsing_method (Callable, optional): A method to parse the AI response.

        Returns:
            Tuple[str, str]: A tuple containing the response message for the human and
                              for the system.

        Raises:
            Exception: If the OpenAI API call fails or the parsing method encounters an error.
        """
        messages_data = []

        try:
            # Send the message to the OpenAI assistant
            await self._send_message_to_assistant(message)
            
            # Initiate the assistant run and wait for completion
            await self._wait_for_assistant_run_completion()

            # Retrieve the latest messages from the assistant
            messages_data = await self._retrieve_messages()

            ai_message_for_human, ai_message_for_system, error = (
                    await self._process_ai_response(messages_data, parsing_method)
                )

            if error:
                if self.retry_count < self.config.max_retries:
                    self.retry_count += 1
                    logger.info(f"Retrying due to error: {error}. Attempt {self.retry_count}")
                    retry_message = f"Error encountered while procesing your response:\n{error}.\nPlease clarify or modify the respose. Pay attention to the JSON format required. Don't forget to extrictly use the same language used with the user. This is not the user. I'm the system and I'll always respond in EN"
                    return await self.request_ai_response(retry_message, parsing_method)
                else:
                    logger.error("Maximum retries reached, unable to resolve the issue.")
                    return "Unfortunately, I couldn't process your request due to repeated errors.", ""

            # Reset retry count on successful processing
            self.retry_count = 0            
            return ai_message_for_human, ai_message_for_system                

        except Exception as e:
            logger.error(
                f"An error occurred during AI response generation: {e}/n/n{traceback.format_exc()}",
                exc_info=True,
            )
            return "", ""

    async def _send_message_to_assistant(self, message: str) -> None:
        """
        Sends the user's message to the OpenAI assistant thread.

        Args:
            message (str): The message to send to the assistant.

        Raises:
            Exception: If sending the message fails.
        """
        await self.client.beta.threads.messages.create(
            thread_id=self.openai_thread.id,
            role="user",
            content=message,
        )
        logger.info(f"Message sent to {self.config.name}: {message}")

    async def _wait_for_assistant_run_completion(self) -> Any:
        """
        Initiates a run with the OpenAI assistant and waits for it to complete.

        Returns:
            Any: The run object after completion.

        Raises:
            Exception: If initiating or waiting for the run fails.
        """
        run = await self.client.beta.threads.runs.create(
            thread_id=self.openai_thread.id,
            assistant_id=self.openai_assistant.id,
        )
        logger.info(
            f"{self.config.name} assistant run initiated, waiting for completion..."
        )
        while run.status in ["queued", "in_progress"]:
            await asyncio.sleep(0.5)  # Non-blocking sleep
            run = await self.client.beta.threads.runs.retrieve(
                thread_id=self.openai_thread.id,
                run_id=run.id,
            )
        logger.info(f"{self.config.name} assistant run completed.")
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
            thread_id=self.openai_thread.id, order="desc"
        )
        logger.info(f"Retrieved messages from {self.config.name}")
        return messages.data

    async def _process_ai_response(
        self, messages_data: List, parsing_method: Optional[Callable] = None
    ) -> Tuple[Optional[str], Optional[WSInput], Optional[str]]:
        """
        Processes the OpenAI assistant's latest response.

        Args:
            messages_data (List): The list of messages data from the assistant.
            parsing_method (Callable, optional): A method to parse the AI response.

        Returns:
            Tuple[str, WSInput, str]: A tuple of the AI's response for the human, the system command and any error that might have occured.

        Raises:
            Exception: If processing the response fails.
        """
        # Process the response to find the latest AI message
        ai_message = next((m for m in messages_data if m.role == "assistant"), None)
        if not ai_message:
            raise Exception("No AI message found.")

        # Ensure that ai_message.content has at least one element
        if not ai_message.content or len(ai_message.content) == 0:
            raise Exception("AI message content is empty.")

        ai_message_content = ai_message.content[0].text.value
        if parsing_method:
            return parsing_method(ai_message_content)
        else:
            return self._default_parsing(ai_message_content)

    @abstractmethod
    def _default_parsing(
        self, ai_message_content: str
    ) -> Tuple[Optional[str], Optional[WSInput], Optional[str]]:
        """
        Robustly parses AI response content, extracting `message_for_human` and creating a WSInput instance for `message_for_system`.
        """
        pass

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
            try:
                with open(file_path, "rb") as file:
                    uploaded_file = await self.client.files.create(
                        file=file, purpose="assistants"
                    )
                    file_ids.append(uploaded_file.id)
            except Exception as e:
                logger.error(f"Could not upload file {file_path} to OpenAI: {e}")
                continue  # Skip this file and proceed with the next iteration

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
            try:
                await self.client.files.delete(file_id)
            except Exception as e:
                logger.info(
                    f"Could not delete file with ID {file_id}: {e}"
                )  # Log or print the error
                continue  # Skip this file and proceed with the next iteration
        self.file_ids = []

        # Then upload the new files
        return await self.create_and_upload_knowledge_files(files_paths)

    async def start_assistant(self, assistant_id=None, files_paths: List[str] = []):
        """
        Starts the chat assistant.
        """
        try:
            self.file_ids = await self.create_and_upload_knowledge_files(files_paths)
            if assistant_id:
                await self._retrieve_assistant(assistant_id)
            else:
                await self._create_assistant()

        except Exception as e:
            logger.error(f"Error initializing assistant: {traceback.format_exc()}")
            raise e

    async def update_assistant(self, files_paths: List[str]):
        """
        Starts the chat assistant.
        """
        try:
            self.file_ids = await self.update_and_upload_knowledge_files(files_paths)
            await self._update_assistant()

        except Exception as e:
            logger.error(f"Error updating assistant: {traceback.format_exc()}")
            raise e
