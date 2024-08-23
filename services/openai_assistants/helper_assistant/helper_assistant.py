import logging
from typing import Optional, Tuple
from openai.types.beta.assistant_response_format_option_param import ResponseFormatText


from services.utils.log_utils import get_logger
from services.openai_assistants.assistant import Assistant
from .helper_assistant_response import HelperAssistantResponse
from .helper_assistant_input import HelperAssistantInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class HelperAssistant(Assistant[HelperAssistantInput, HelperAssistantResponse]):

    def __init__(self, config):
        """
        Initialize the Helper Assistant.

        Args:
            config (DictConfig): Configuration parameters.
        """
        super().__init__(config)

    def _initialize_response_format(self):
        """
        Initialize the response format specific to ChatAssistant.
        """
        self._response_format = ResponseFormatText(type="text")

    def _default_parsing(self, ai_message_content: str) -> HelperAssistantResponse:
        return HelperAssistantResponse(message_for_user=ai_message_content, error=None)

    def _default_error_processing_request(
        self, message: str, error: str
    ) -> HelperAssistantResponse:
        return HelperAssistantResponse(
            message_for_user=message,
            error=error,
        )
