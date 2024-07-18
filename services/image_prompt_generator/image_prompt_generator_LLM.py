import logging
from abc import ABC
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Any

from langchain_community.callbacks import get_openai_callback
from langchain_community.callbacks.openai_info import OpenAICallbackHandler

from langchain_community.chat_models.openai import BaseChatModel
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseMessage, OutputParserException

from services.prompts_constructors.utils.prompt_utils import get_retry_output_parser

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class ImagePromptGeneratorLLM(ABC):
    """
    Abstract base class for story-related LLM Machines (Generation and Critic).
    """

    def __init__(
        self,
        main_llm: BaseChatModel,
        parser_llm: BaseChatModel,
        prompt_constructor: ModuleType,
        num_outputs: Optional[int] = 1,
    ):
        """
        Initialize the ChapterBaseLLM.

        Args:
            main_llm (BaseChatModel): The primary LLM used for generating responses.
            parser_llm (BaseChatModel): The secondary LLM used for parsing incorrect responses.
            prompt_constructor (ModuleType): The module used for rendering prompts.
            num_outputs (int, optional): The number of outputs to generate. Defaults to 1.
        """
        self.main_llm = main_llm
        self.parser_llm = parser_llm
        self.prompt_constructor = prompt_constructor
        self.num_outputs = num_outputs

        self.total_tokens: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_cost: float = 0.0
        self.successful_requests: int = 0

    def _track_cost(self, cb: OpenAICallbackHandler):
        """
        Track the cost of using the LLM.

        Args:
            cb (OpenAICallbackHandler): The callback handler.
        """
        self.total_tokens += cb.total_tokens
        self.prompt_tokens += cb.prompt_tokens
        self.completion_tokens += cb.completion_tokens
        self.total_cost += cb.total_cost
        self.successful_requests += cb.successful_requests

    def generate_image_prompts(
        self, chapter_number: int, chapter_content: str, is_cover: bool = False
    ) -> Tuple[List[Tuple[bool, Dict[str, Any]]], List[Any]]:        
        """
        Generates output from the LLM.

        Args:
            input_info (Dict[str, str]): The latest info about the state of the world.

        Returns:
            Tuple[List[Tuple[bool, Dict[str, str]]], List[BaseMessage]]: A tuple of outputs and messages.
            Output artifacts is a list of tuples, each containing a status boolean and an output dict.
            Messages is a list of prompts that were constructed to generate output.
        """        
        # Prepare prompts and parser
        (
            system_message,
            output_parser,
        ) = self.prompt_constructor.compose_system_message_and_create_output_parser()

        human_message = self.prompt_constructor.compose_message_for_LLM(chapter_number, chapter_content, is_cover)
        messages = [system_message, human_message]

        # Generate output
        output_artifacts = []
        for _ in range(self.num_outputs):
            with get_openai_callback() as cb:
                ai_message = self.main_llm(messages)
                retry_output_parser = get_retry_output_parser(
                    output_parser, self.parser_llm
                )
                # Track cost for the main llm (producing responses)
                self._track_cost(cb)

            with get_openai_callback() as cb:
                # Parse an output from LLM
                try:
                    chat_prompt_value = ChatPromptTemplate.from_messages(
                        messages
                    ).format_prompt()
                    output = retry_output_parser.parse_with_prompt(
                        ai_message.content, chat_prompt_value
                    )
                except OutputParserException as ex:
                    logger.exception("Could not parse llm output")
                    output_artifacts.append((False, dict()))
                else:
                    output_artifacts.append((True, output))

                # Track cost for the parser llm (solving parsing problems with new requests to the an llm)
                self._track_cost(cb)

        return output_artifacts, messages
