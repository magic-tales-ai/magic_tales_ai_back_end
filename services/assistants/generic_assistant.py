from abc import ABC, abstractmethod
import os
import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Callable
import traceback

from pydantic import BaseModel
from langchain_community import llms
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from services.utils.log_utils import get_logger
from services.assistants.assistant_response.assistant_response import (
    AssistantResponse,
)
from services.assistants.assistant_input.assistant_input import (
    AssistantInput,
    Source,
)
from services.assistants.prompt_utils import (
    load_prompt_template_from_file,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

TInput = TypeVar("TInput", bound=AssistantInput)
TResponse = TypeVar("TResponse", bound=AssistantResponse)


class Assistant(ABC, Generic[TInput, TResponse]):
    def __init__(self, config: Dict):
        """
        Initialize the Assistant.

        Args:
            config (Dict): Configuration parameters.
        """
        self.config = config
        self.instructions = None
        self.vector_store = None
        self.latest_messages_data = []
        self.retry_count = 0

        self.llm = self._initialize_llm()

        logger.info(f"{self.config['name']} AI assistant class initialized.")

    def _initialize_llm(self):
        """
        Initialize the LLM based on configuration.

        Returns:
            LLM: The initialized language model.
        """
        try:
            if not self.config.instructions_path:
                raise ValueError(
                    "Instructions path is required for assistant creation."
                )

            self.instructions = SystemMessage(
                content=load_prompt_template_from_file(self.config.instructions_path)
            )

            model_type = self.config.get("model_type", "openai")
            if model_type == "openai":
                return ChatOpenAI(
                    model=self.config.model,
                    name=self.config.name,
                    temperature=self.config.temperature,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    # instructions=instructions,
                    # tools=self.config.tools,
                    # tool_resources=tool_resources,
                    # response_format=self._response_format,
                )
            elif model_type == "claude":
                return ChatAnthropic(
                    model=self.config.model,
                    name=self.config.name,
                    temperature=self.config.temperature,
                    api_key=os.getenv("ANTHROPIC_API_KEY"),
                )
            elif model_type == "ollama":
                return OllamaLLM(
                    model=self.config.model,
                    temperature=self.config.temperature,
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        except Exception as e:
            logger.error(f"Error creating assistant: {traceback.format_exc()}")
            raise
    

    async def _create_vector_store(self) -> None:
        """
        Create and initialize a vector store.
        """
        embedding_model = self._initialize_embedding_model()
        vector_store_type = self.config.vector_store_type

        if vector_store_type == "faiss":
            self.vector_store = FAISS(embedding_model)
        elif vector_store_type == "chroma":
            self.vector_store = Chroma(embedding_function=embedding_model)
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")

        logger.info(f"Vector store initialized with type: {vector_store_type}")

    def _initialize_embedding_model(self):
        """
        Initialize the embedding model based on configuration.

        Returns:
            Embeddings: The initialized embedding model.
        """
        embedding_type = self.config.embedding_type
        if embedding_type == "openai":
            return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        elif embedding_type == "huggingface":
            return HuggingFaceEmbeddings()
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

    async def _load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load and embed documents from a list of file paths, without requiring specific schema knowledge.

        Args:
            file_paths (List[str]): List of paths to the documents to load and embed.

        Returns:
            List[Document]: A list of Document objects created from the files' contents.
        """
        all_documents = []
        try:
            for file_path in file_paths:
                try:
                    # Load the document
                    document = await self._extract_text_from_file(file_path)

                    # Handle large documents by chunking
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(document)

                    # Append the split documents to the all_documents list
                    all_documents.extend(splits)
                    
                except Exception as e:
                    logger.error(f"Could not load file {file_path}: {e}")
                    continue

            return all_documents
        except Exception as e:
            logger.error(f"Error in loading documents: {e}")
            raise

    async def _extract_text_from_file(self, file_path: str) -> Document:
        """
        Extract text from a file and return a Langchain Document.

        Args:
            file_path (str): The path to the file.

        Returns:
            Document: A Langchain Document object containing the text content and metadata.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Create a Document object with the content and metadata
            document = Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_size": os.path.getsize(file_path),
                    "file_type": os.path.splitext(file_path)[-1]
                }
            )
            return document

        except Exception as e:
            logger.error(f"Error extracting text from file {file_path}: {e}")
            raise

    async def _upload_knowledge_files(self, file_paths: List[str]) -> None:
        """
        Creates and Populates the vector store with provided documents.

        Args:
            file_paths (List[str]): Paths to the knowledge base files.
        """
        if not self.vector_store:
            await self._create_vector_store()

        try:
            self.vector_store.delete_collection()

            # Add documents to the vector store
            documents = await self._load_documents(file_paths)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)

            await self.vector_store.aadd_documents(documents=splits)
        except Exception as e:
            logger.error(
                f"Error in uploading knowledge files into '{self.config.name}' assistant's vector store: {e}"
            )
            raise        

    async def _create_and_populate_vector_store(self, file_paths: List[str]) -> None:
        """
        Creates a vector store and populates it with documents.

        Args:
            file_paths (List[str]): Paths to the knowledge base files.
        """
        await self._create_vector_store()
        if file_paths:
            await self._upload_knowledge_files(file_paths)

    async def request_ai_response(
        self,
        message_for_assistant: TInput,
        parsing_method: Optional[Callable] = None,
    ) -> TResponse:
        """
        Process the incoming message and generate an AI response.

        Args:
            message_for_assistant (TInput): The input for the assistant containing the message and its source.
            parsing_method (Callable, optional): A method to parse the AI response.

        Returns:
            TResponse

        Raises:
            Exception: If the LLM API call fails or the parsing method encounters an error.
        """
        try:
            prompt = await self._build_prompt(message_for_assistant)
            ai_response = await self.llm.agenerate(prompt)
            logger.info(f"Raw AI Respose from {self.config.name}:{ai_response}")

            response = await self._process_ai_response(ai_response, parsing_method)

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

    async def _process_ai_response(
        self, ai_response: str, parsing_method: Optional[Callable] = None
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
        # Parse the response
        if parsing_method:
            return parsing_method(ai_response)
        else:
            return self._default_parsing(ai_response)

    async def _build_prompt(self, message_for_assistant: TInput) -> str:
        """
        Build the prompt using the message for the assistant.

        Args:
            message_for_assistant (TInput): The input message to the assistant.

        Returns:
            str: The constructed prompt.
        """
        message_to_assistant = json.dumps(await message_for_assistant.to_json())
        full_message_for_assistant = ChatPromptTemplate.from_messages([
            self.instructions,
            ("human", message_to_assistant)
        ])
        logger.info(f"Message sent to {self.config.name}: {full_message_for_assistant}")
        return full_message_for_assistant

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

    async def start_assistant(self, files_paths: List[str] = []):
        """
        Starts the chat assistant.

        Args:
            documents (List[str]): Optional list of documents to populate the vector store with.
        """
        logger.info(f"Starting {self.config['name']} assistant ...")
        try:
            await self._create_and_populate_vector_store(files_paths)

        except Exception as e:
            logger.error(f"Error initializing assistant: {e}")
            raise e
