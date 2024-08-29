import logging
from abc import ABC
from types import ModuleType
from typing import Dict, List, Optional, Tuple, Union, Any

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM

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
        main_llm: Union[ChatOpenAI, ChatAnthropic, OllamaLLM],
        parser_llm: Union[ChatOpenAI, ChatAnthropic, OllamaLLM],
        prompt_constructor: ModuleType,        
        num_outputs: Optional[int] = 1,
    ):
        """
        Initialize the ChapterBaseLLM.

        Args:
            main_llm (ChatOpenAI): The primary LLM used for generating responses.
            parser_llm (ChatOpenAI): The secondary LLM used for parsing incorrect responses.
            prompt_constructor (ModuleType): The module used for rendering prompts.
            num_outputs (int, optional): The number of outputs to generate. Defaults to 1.
        """
        self.main_llm = main_llm
        self.parser_llm = parser_llm
        self.prompt_constructor = prompt_constructor
        self.num_outputs = num_outputs

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

        human_message = self.prompt_constructor.compose_message_for_LLM(
            chapter_number, chapter_content, is_cover
        )
        messages = [system_message, human_message]

        # Generate output
        output_artifacts = []
        for _ in range(self.num_outputs):
            ai_message = self.main_llm(messages)
            retry_output_parser = get_retry_output_parser(
                output_parser, self.parser_llm
            )
            try:
                chat_prompt_value = ChatPromptTemplate.from_messages(
                    messages
                ).format_prompt()
                output = retry_output_parser.parse_with_prompt(
                    ai_message.content, chat_prompt_value
                )
                output_artifacts.append((True, output))

            except OutputParserException as ex:
                logger.exception("Could not parse llm output")
                output_artifacts.append((False, dict()))

        return output_artifacts, messages
