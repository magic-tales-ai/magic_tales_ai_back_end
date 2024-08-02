from typing import Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage

from services.prompts_constructors.critic import prompt_constructor

from .chapter_base_LLM import ChapterBaseLLM

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class ChapterCriticLLM(ChapterBaseLLM):
    """Agent that evaluates and critiques the performance of a story."""

    def __init__(
        self,
        main_llm: ChatOpenAI,
        parser_llm: ChatOpenAI,
        story_blueprint: Dict[str, str],
        previous_chapter_content: str,
        num_outputs: Optional[int] = 1,
    ):
        """
        Initialize the ChapterCriticLLM.

        Args:
            main_llm (ChatOpenAI): The LLM used for criticizing story.
            parser_llm (ChatOpenAI): The LLM used for parsing incorrect responses.
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

    def generate_critique(
        self, input_info: Dict[str, str]
    ) -> Tuple[bool, Dict[str, str], List[BaseMessage]]:
        """
        Generate a critique for a given story.

        Args:
            input_info (Dict[str, str]): Information about the input that needs critique.

        Returns:
            Tuple[bool, Dict[str, str], List[BaseMessage]]: Tuple containing a status boolean, the critique output,
            and a list of messages.
        """
        logger.info("Start critique generation, using LLM.")
        return self._generate_output(input_info)
