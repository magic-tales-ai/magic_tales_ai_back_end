from typing import Any, Dict, List, Optional, Tuple
import traceback
import json
import logging
import re


from services.utils.log_utils import get_logger
from services.openai_assistants.assistant import Assistant
from services.custom_exceptions.custom_exceptions import NotADictionaryError
from .chat_assistant_response import ChatAssistantResponse, StructuredResponse
from .chat_assistant_input import ChatAssistantInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class JSONParsingError(Exception):
    pass


class ChatAssistant(Assistant[ChatAssistantInput, ChatAssistantResponse]):
    def __init__(self, config):
        """
        Initialize the Chat Assistant.

        Args:
            config (DictConfig): Configuration parameters.
        """
        super().__init__(config)

    def _initialize_response_format(self):
        """
        Initialize the response format specific to ChatAssistant.
        """
        self._response_format = StructuredResponse

    def _parse_with_fallbacks(self, ai_message_content: str) -> ChatAssistantResponse:
        """
        Attempts to parse AI message content as JSON and extract required fields.

        This method uses a series of fallback strategies to handle various input formats
        and potential issues, aiming to always return a valid ChatAssistantResponse.

        Args:
            ai_message_content (str): The raw AI response content.

        Returns:
            ChatAssistantResponse: Containing extracted information or error details.
        """

        def _safe_json_loads(content: str) -> Dict[str, Any]:
            """Safely attempt to parse JSON content."""
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed: {e}")
                raise JSONParsingError(f"Invalid JSON: {e}")

        def _extract_json_from_codeblock(content: str) -> str:
            """Extract JSON content from a markdown code block if present."""
            import re

            json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            match = re.search(json_pattern, content)
            return match.group(1) if match else content

        def _sanitize_content(content: str) -> str:
            """Remove potential problematic characters from the content."""
            return content.strip().replace("\n", "").replace("\r", "")

        def _validate_message_for_system(msg: Any) -> Dict[str, Any]:
            """Ensure message_for_system is a dictionary."""
            if not isinstance(msg, dict):
                logger.warning(
                    "message_for_system is not a dictionary. Converting to empty dict."
                )
                return {}
            return msg

        def _create_error_response(error_msg: str) -> ChatAssistantResponse:
            """Create an error response with the original content as message_for_user."""
            return ChatAssistantResponse(
                message_for_user=ai_message_content,
                message_for_system=None,
                user_language=None,
                error=error_msg,
            )

        try:
            # Extract JSON if it's within a code block
            content = _extract_json_from_codeblock(ai_message_content)

            # Sanitize the content
            sanitized_content = _sanitize_content(content)

            # Parse the JSON
            parsed_content = _safe_json_loads(sanitized_content)

            # Extract and validate fields
            message_for_user = parsed_content.get("message_for_user")
            message_for_system = _validate_message_for_system(
                parsed_content.get("message_for_system")
            )
            user_language = parsed_content.get("user_language")

            return ChatAssistantResponse(
                message_for_user=message_for_user,
                message_for_system=message_for_system,
                user_language=user_language,
                error=None,
            )

        except JSONParsingError as e:
            return _create_error_response(f"JSON parsing failed: {str(e)}")
        except Exception as e:
            logger.error(
                f"Unexpected error in _parse_with_fallbacks: {str(e)}", exc_info=True
            )
            return _create_error_response(f"An unexpected error occurred: {str(e)}")

    def _default_parsing(self, ai_message_content: str) -> ChatAssistantResponse:
        """
        Robustly parses AI response content, extracting `message_for_user`, creating a dict instance for `message_for_system AND pass over the error if it occurs.
        """
        return self._parse_with_fallbacks(ai_message_content)

    def _default_error_processing_request(
        self, message: str, error: str
    ) -> ChatAssistantResponse:
        return ChatAssistantResponse(
            message_for_user=message,
            message_for_system={},
            user_language=None,
            error=error,
        )
