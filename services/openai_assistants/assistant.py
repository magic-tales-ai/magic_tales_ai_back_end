from abc import ABC, abstractmethod
import traceback
import os
import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable

from openai import AsyncOpenAI
from openai.types.beta.assistant_create_params import (
    ToolResources,
    ToolResourcesFileSearch,
    ToolResourcesCodeInterpreter,
)

from services.utils.log_utils import get_logger
from services.openai_assistants.assistant_response.assistant_response import (
    AssistantResponse,
)
from services.openai_assistants.assistant_input.assistant_input import (
    AssistantInput,
    Source,
)
from services.openai_assistants.prompt_utils import (
    async_load_prompt_template_from_file,
)
from magic_tales_models.models.ws_input import WSInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

TInput = TypeVar("TInput", bound=AssistantInput)
TResponse = TypeVar("TResponse", bound=AssistantResponse)


class Assistant(ABC, Generic[TInput, TResponse]):
    def __init__(self, config: Dict):
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
        self.vector_store_id = None
        self.file_ids = []
        self.latest_messages_data = []
        self.retry_count = 0

        logger.info(f"{self.config.name} AI assistant class initialized.")

    def _validate_openai_api_key(self):
        """Validates the presence of the OpenAI API key."""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    async def _create_vector_store(self) -> str:
        """
        Create a vector store to handle multiple files.

        Returns:
            str: The ID of the created vector store.
        """
        try:
            vector_store = await self.client.beta.vector_stores.create(
                name=f"{self.config.name}_vector_store"
            )
            logger.info(f"Vector store created with ID: {vector_store.id}")
            return vector_store.id
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    async def _create_tool_resources(self) -> Optional[ToolResources]:
        """
        Creates ToolResources with a vector store for file search and code interpreter tools.

        Returns:
            Optional[ToolResources]: The created tool resources, or None if not applicable.
        """
        tool_resources: ToolResources = {}

        try:
            vector_store_ids = await self._create_and_populate_vector_store()

            if vector_store_ids:
                if self._is_tool_configured("file_search"):
                    tool_resources["file_search"] = ToolResourcesFileSearch(
                        vector_store_ids=vector_store_ids
                    )

                if self._is_tool_configured("code_interpreter"):
                    tool_resources["code_interpreter"] = ToolResourcesCodeInterpreter(
                        file_ids=self.file_ids
                    )

        except Exception as e:
            logger.exception(f"Error creating tool resources: {e}")
            return None

        return tool_resources if tool_resources else None

    async def _create_and_populate_vector_store(self) -> List[str]:
        """
        Creates a vector store and attaches the specified files to it.

        Returns:
            List[str]: The list of vector store IDs, or an empty list if no files are provided.
        """
        if not self.file_ids:
            logger.warning(
                f"File IDs are empty for {self.config.name} assistant, skipping vector store creation."
            )
            return []

        try:
            vector_store_id = await self._create_vector_store()
            await self._attach_files_to_vector_store(vector_store_id)
            return [vector_store_id]
        except Exception as e:
            logger.exception(f"Error creating and populating vector store: {e}")
            return []

    def _is_tool_configured(self, tool_type: str) -> bool:
        """
        Checks if a specific tool type is configured in the assistant's configuration.

        Args:
            tool_type (str): The type of the tool to check.

        Returns:
            bool: True if the tool is configured, False otherwise.
        """
        return self.config.tools and any(
            tool.get("type") == tool_type for tool in self.config.tools
        )

    async def _attach_files_to_vector_store(self, vector_store_id: str) -> None:
        """
        Attach files to the specified vector store.

        Args:
            vector_store_id (str): The ID of the vector store.
        """
        try:
            await self.client.beta.vector_stores.file_batches.create(
                vector_store_id=vector_store_id, file_ids=self.file_ids
            )
            logger.info("Files attached to vector store successfully.")
        except Exception as e:
            logger.error(f"Error attaching files to vector store: {e}")
            raise

    async def _create_assistant(self):
        """
        Creates a NEW OpenAI assistant with attached knowledge base files.
        """
        try:
            if not self.config.instructions_path:
                raise ValueError(
                    "Instructions path is required for assistant creation."
                )

            instructions = await async_load_prompt_template_from_file(
                self.config.instructions_path
            )
            tool_resources = await self._create_tool_resources()

            if not tool_resources and self.file_ids:
                raise ValueError(
                    "Failed to create tool resources with provided file IDs."
                )

            self.openai_assistant = await self.client.beta.assistants.create(
                model=self.config.model,
                name=self.config.name,
                instructions=instructions,
                temperature=self.config.temperature,
                tools=self.config.tools,
                tool_resources=tool_resources,
            )

            self.openai_thread = await self.client.beta.threads.create()
            logger.info(f"{self.config.name} OpenAI Assistant Created.")
        except Exception as e:
            logger.error(f"Error creating assistant: {traceback.format_exc()}")
            raise

    async def _retrieve_assistant(self, assistant_id=None):
        """
        Retrieves an existing OpenAI assistant with attached knowledge base files.
        """
        try:
            if not assistant_id:
                logger.warning(
                    "Assistant ID is required for retrieval!! We will create a new Assistant."
                )
                await self._create_assistant()
                return

            instructions = await async_load_prompt_template_from_file(
                self.config.instructions_path
            )
            tool_resources = await self._create_tool_resources()

            if not tool_resources and self.file_ids:
                raise ValueError(
                    "Failed to create tool resources with provided file IDs."
                )

            self.openai_assistant = await self.client.beta.assistants.update(
                assistant_id=assistant_id,
                model=self.config.model,
                name=self.config.name,
                instructions=instructions,
                tools=self.config.tools,
                tool_resources=tool_resources,
            )
        except Exception as e:
            logger.error(
                f"Failed to retrieve assistant ID: {assistant_id} from OpenAI. Creating a new Assistant. Error: {e}"
            )
            await self._create_assistant()
            return

        self.openai_thread = await self.client.beta.threads.create()
        logger.info(f"{self.config.name} OpenAI Assistant retrieved successfully.")
        return

    async def _update_assistant(self):
        """
        Updates the OpenAI assistant with attached knowledge base files.
        """
        try:
            if not self.openai_assistant or not self.openai_assistant.id:
                raise ValueError("Assistant ID is required for updating.")

            tool_resources = await self._create_tool_resources()

            if not tool_resources and self.file_ids:
                raise ValueError(
                    "Failed to create tool resources with provided file IDs."
                )

            self.openai_assistant = await self.client.beta.assistants.update(
                assistant_id=self.openai_assistant.id,
                tools=self.config.tools,
                tool_resources=tool_resources,
            )
            logger.info(f"{self.config.name} OpenAI Assistant UPDATED.")
        except Exception as e:
            logger.error(f"Error updating assistant: {traceback.format_exc()}")
            raise

    async def request_ai_response(
        self,
        message_for_assistant: TInput,
        parsing_method: Optional[Callable] = None,
    ) -> TResponse:
        """
        Process the incoming message and generate an AI response.

        The function sends the message to the OpenAI assistant and retrieves the response.
        It handles the entire process including sending the message, initiating the run, waiting
        for completion, and processing the received response.

        Args:
            message_for_assistant (TInput): The input for the assistant containing the message and its source.
            parsing_method (Callable, optional): A method to parse the AI response.

        Returns:
            TResponse

        Raises:
            Exception: If the OpenAI API call fails or the parsing method encounters an error.
        """
        try:
            # Send the message to the OpenAI assistant
            await self._send_message_to_assistant(
                json.dumps(await message_for_assistant.to_json())
            )

            # Initiate the assistant run and wait for completion
            await self._wait_for_assistant_run_completion()

            # Retrieve the latest messages from the assistant
            self.latest_messages_data = await self._retrieve_messages()

            response = await self._process_ai_response(
                self.latest_messages_data, parsing_method
            )

            if await response.has_error():
                if self.retry_count < self.config.max_retries:
                    self.retry_count += 1
                    logger.warn(
                        f"Retrying due to error: {response.error}. Attempt {self.retry_count}"
                    )
                    retry_message = message_for_assistant.__class__(
                        message=f"Your JSON response:\n{response.message_for_user}\n It was very likely incorrectly formated and it threw the following error:\n{response.error}.\nPlease adjust you response so that this call 'json.loads(your_json_response)' DOES NOT fail again. Test it and validate it using your tools before sending it.",
                        source=Source.SYSTEM,
                    )

                    return await self.request_ai_response(retry_message, parsing_method)
                else:
                    logger.error(
                        "Maximum retries reached, unable to resolve the issue."
                    )
                    self.retry_count = 0
                    return self._default_error_processing_request(
                        message="Unfortunately, I couldn't process your request. Let's try again in a second.",
                        error="Generic error or maximum retries parsing reached, issue unresolved.",
                    )

            # Reset retry count on successful processing
            self.retry_count = 0

            return response

        except Exception as e:
            error = str(e)  # traceback.format_exc(e)
            logger.error(f"An error occurred during AI response generation: {error}")
            return self._default_error_processing_request(
                message="I apologize. An error has occurred, please try again. :-(",
                error=error,
            )

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
    ) -> TResponse:
        """
        Processes the OpenAI assistant's latest response.

        Args:
            messages_data (List): The list of messages data from the assistant.
            parsing_method (Callable, optional): A method to parse the AI response.

        Returns:
            TResponse

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
    def _default_parsing(self, ai_message_content: str) -> TResponse:
        """
        Define how each assistant type should parse the AI's response.
        This method needs to be implemented by each subclass.
        """
        pass

    @abstractmethod
    def _default_error_processing_request(self, message: str, error: str) -> TResponse:
        """
        Define how each assistant type should parse the AI's response.
        This method needs to be implemented by each subclass.
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
        logger.info(
            f"Createing and uploading knowledge files for {self.config.name} assistant ..."
        )
        try:
            files_paths = files_paths or []
            file_ids = []
            for file_path in files_paths:
                try:
                    with open(file_path, "rb") as file:
                        file_obj = await self.client.files.create(
                            file=file, purpose="assistants"
                        )
                        file_ids.append(file_obj.id)
                        logger.info(
                            f"File '{file_path}' created successfully using OpenAI API."
                        )
                except Exception as e:
                    logger.error(f"Could not upload file {file_path} to OpenAI: {e}")
                    continue

            return file_ids
        except Exception as e:
            logger.error(f"Error in creating and uploading knowledge files: {e}")
            raise

    async def update_and_upload_knowledge_files(
        self, files_paths: List[str]
    ) -> List[str]:
        """
        Updates the changed files and uploads knowledge base files to OpenAI and returns their IDs.

        Args:
            files_paths (List[str]): Paths to the new knowledge base files.

        Returns:
            List[str]: A list of file IDs after upload.
        """
        logger.info(
            f"Updating and uploading knowledge files for {self.config.name} assistant ..."
        )
        try:
            for file_id in self.file_ids:
                try:
                    await self.client.files.delete(file_id)
                except Exception as e:
                    logger.info(f"Could not delete file with ID {file_id}: {e}")
                    continue
            deleted_vector_store = self.client.beta.vector_stores.delete(
                vector_store_id=self.vector_store_id
            )
            self.vector_store_id = None
            return await self.create_and_upload_knowledge_files(files_paths)
        except Exception as e:
            logger.error(f"Error updating and uploading knowledge files: {e}")
            raise

    async def start_assistant(self, assistant_id=None, files_paths: List[str] = []):
        """
        Starts the chat assistant.
        """
        logger.info(f"Starting {self.config.name} assistant ...")
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
        logger.info(f"Updating {self.config.name} assistant ...")
        try:
            self.file_ids = await self.update_and_upload_knowledge_files(files_paths)
            await self._update_assistant()

        except Exception as e:
            logger.error(f"Error updating assistant: {traceback.format_exc()}")
            raise e
