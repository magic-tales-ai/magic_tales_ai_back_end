from typing import Any, Dict, List, Optional, Tuple
import traceback
import json
import logging
from openai.types.beta.assistant_response_format_option_param import ResponseFormatText


from services.utils.log_utils import get_logger
from services.openai_assistants.assistant import Assistant
from .supervisor_assistant_response import SupervisorAssistantResponse
from .supervisor_assistant_input import SupervisorAssistantInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class SupervisorAssistant(
    Assistant[SupervisorAssistantInput, SupervisorAssistantResponse]
):
    def __init__(self, config):
        """
        Initialize the Supervisor Assistant.

        Args:
            config (DictConfig): Configuration parameters.
        """
        super().__init__(config)
        self.interventions_count = 0
        self.interventions_count_limit = config.interventions_count_limit

    def _initialize_response_format(self):
        """
        Initialize the response format specific to ChatAssistant.
        """
        self._response_format = ResponseFormatText(type="text")

    def _parse_with_fallbacks(
        self, ai_message_content: str
    ) -> SupervisorAssistantResponse:
        """
        Attempts to directly parse AI message content as JSON and extract `message_for_user` and `message_for_system`.

        Args:
            ai_message_content (str): The raw AI response content.

        Returns:
            A SupervisorAssistantResponse containing: `message_for_user` as str and `message_for_system` as a dict or None for each if not applicable or errors occur.
        """
        try:
            # # Remove surrounding characters and escape single quotes
            # sanitized_content = ai_message_content.strip("\`\n").replace("'", "\\'")

            # # Remove leading and trailing quotes
            # sanitized_content = sanitized_content.strip('"')

            # # Replace escaped double quotes with single quotes
            # sanitized_content = sanitized_content.replace('\\"', '"')

            sanitized_content = ai_message_content

            # Parse the sanitized JSON
            ai_message_dict = json.loads(sanitized_content)

            # Extract the required keys
            intervention_needed = ai_message_dict.get("intervention_needed", "")
            intervention_message = ai_message_dict.get("intervention_message", "")

            # Ensure message_for_system is a dictionary
            if not isinstance(intervention_message, str):
                logger.warning(
                    "message_for_system is not a dictionary. Resetting to empty dict."
                )
                message_for_system = {}

            return SupervisorAssistantResponse(
                intervention_needed=intervention_needed,
                message_for_user=intervention_message,
                error=None,
            )

        except (json.JSONDecodeError, KeyError) as e:
            error = traceback.format_exc()
            logger.warning(f"JSON parsing failed, attempting regex extraction: {error}")
            # return self._extract_with_regex(ai_message_content)
            return SupervisorAssistantResponse(
                intervention_needed=None,
                message_for_user=sanitized_content,
                error=e,
            )

    def _default_parsing(self, ai_message_content: str) -> SupervisorAssistantResponse:
        """
        Robustly parses AI response content, extracting `message_for_user`, creating a dict instance for `message_for_system AND pass over the error if it occurs.
        """
        return self._parse_with_fallbacks(ai_message_content)

    def _default_error_processing_request(
        self, message: str, error: str
    ) -> SupervisorAssistantResponse:
        return SupervisorAssistantResponse(
            intervention_needed=None,
            message_for_user=message,
            error=error,
        )
