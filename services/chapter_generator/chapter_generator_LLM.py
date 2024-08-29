from typing import Dict, List, Optional, Tuple


from langchain.schema import BaseMessage

from .chapter_base_LLM import ChapterBaseLLM

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class ChapterGeneratorLLM(ChapterBaseLLM):
    """
    Agent that generates and processes story-based actions using a Large Language Model (LLM).
    """

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
