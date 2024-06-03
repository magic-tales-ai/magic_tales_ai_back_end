import logging
from typing import Optional, Tuple


from services.utils.log_utils import get_logger
from services.openai_assistants.assistant import Assistant
from .helper_assistant_response import HelperAssistantResponse
from .helper_assistant_input import HelperAssistantInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class HelperAssistant(Assistant[HelperAssistantInput, HelperAssistantResponse]):

    def _default_parsing(self, ai_message_content: str) -> HelperAssistantResponse:
        return HelperAssistantResponse(message_for_user=ai_message_content, error=None)

    def _default_error_processing_request(
        self, message: str, error: str
    ) -> HelperAssistantResponse:
        return HelperAssistantResponse(
            message_for_user=message,
            error=error,
        )
