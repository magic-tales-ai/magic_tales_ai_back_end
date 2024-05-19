import logging
from typing import Optional, Tuple


from services.utils.log_utils import get_logger
from services.openai_assistants.assistant import Assistant

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class HelperAssistant(Assistant):

    def _default_parsing(
        self, ai_message_content: str
    ) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
        """
        Robustly parses AI response content, extracting `message_for_human`, creating a WSInput instance for `message_for_system AND pass over the error if it occurs.
        """

        return ai_message_content, None, None
