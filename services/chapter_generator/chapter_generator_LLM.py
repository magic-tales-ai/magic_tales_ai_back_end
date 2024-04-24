from typing import Dict, List, Optional, Tuple

from langchain_community.chat_models.openai import BaseChatModel
from langchain.schema import BaseMessage

from services.prompts_constructors.generator import prompt_constructor

from .chapter_base_LLM import ChapterBaseLLM

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class ChapterGeneratorLLM(ChapterBaseLLM):
    """
    Agent that generates and processes story-based actions using a Large Language Model (LLM).
    """

    def __init__(
        self,
        main_llm: BaseChatModel,
        parser_llm: BaseChatModel,
        story_blueprint: Dict[str, str],
        previous_chapter_content: str,
        num_outputs: Optional[int] = 1,
    ):
        """
        Initialize the ChapterGeneratorLLM.

        Args:
            main_llm (BaseChatModel): The LLM used for generating story.
            parser_llm (BaseChatModel): The LLM used for parsing incorrect responses.
            num_outputs (int, optional): The number of outputs to generate. Defaults to 1.
        """
        super().__init__(
            main_llm,
            parser_llm,
            prompt_constructor,
            story_blueprint,
            previous_chapter_content,
            num_outputs,
        )

    def generate_chapter_content(
        self, input_info: Dict[str, str]
    ) -> Tuple[List[Tuple[bool, Dict[str, str]]], List[BaseMessage]]:
        """
        Generate Chapter content using LLM.

        Args:
            input_info (Dict[str, str]): Information about the input that needs story generation.

        Returns:
            Tuple[List[Tuple[bool, Dict[str, str]]], List[BaseMessage]]: Tuple containing a list of tuples (each containing a status boolean and the output) and a list of BaseMessage objects.
        """
        logger.info("Requesting chapter content from LLM.")
        return self._generate_output(input_info)
