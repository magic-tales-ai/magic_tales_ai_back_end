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
