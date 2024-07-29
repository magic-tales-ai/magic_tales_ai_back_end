from typing import Any, Dict, List, Optional, Tuple
import traceback
import json
import logging
import re


from services.utils.log_utils import get_logger
from services.openai_assistants.magic_tales_assistant import MagicTalesAgent
from services.custom_exceptions.custom_exceptions import NotADictionaryError
from .chat_assistant_response import ChatAssistantResponse
from .chat_assistant_input import ChatAssistantInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

class ChatAssistant(MagicTalesAgent[ChatAssistantInput, ChatAssistantResponse]):
#class ChatAssistant(Assistant[ChatAssistantInput, ChatAssistantResponse]):
    def __init__(self, config):
        """
        Initialize the Chat Assistant.

        Args:
            config (DictConfig): Configuration parameters.
        """
        super().__init__(config)

    def _parse_with_fallbacks(self, ai_message_content: str) -> ChatAssistantResponse:
        """
        Attempts to directly parse AI message content as JSON and extract `message_for_user` and `message_for_system`.

        Args:
            ai_message_content (str): The raw AI response content.

        Returns:
            A ChatAssistantResponse containing: `message_for_user` as str and `message_for_system` as a dict or None for each if not applicable or errors occur.
        """
        try:
            error_suffix = "Please take a thorough look at your instructions and the exact JSON structure that your responses must have."

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
            message_for_user = ai_message_dict.get("message_for_user")
            message_for_system = ai_message_dict.get("message_for_system")
            user_language = ai_message_dict.get("user_language")

            # Ensure message_for_system is a dictionary
            if not isinstance(message_for_system, dict):
                message_for_system = {}
                raise NotADictionaryError(
                    "message_for_system must be a dictionary. Follow your instructions and the JSON format for your responses"
                )

            return ChatAssistantResponse(
                message_for_user=message_for_user,
                message_for_system=message_for_system,
                user_language=user_language,
                error=None,
            )

        except json.JSONDecodeError as e:
            error = traceback.format_exc()
            logger.error(f"JSON parsing failed: {error}")
            # return self._extract_with_regex(ai_message_content)
            return ChatAssistantResponse(
                message_for_user=sanitized_content,
                message_for_system=None,
                user_language=None,
                error=f"JSON parsing failed: {e}.\n{error_suffix}",
            )
        except KeyError as e:
            error = traceback.format_exc()
            logger.error(f"KeyError: {error}")
            return ChatAssistantResponse(
                message_for_user=sanitized_content,
                message_for_system=None,
                user_language=None,
                error=f"{e}. {error_suffix}",
            )
        except NotADirectoryError as e:
            logger.error("f{e}")
            return ChatAssistantResponse(
                message_for_user=sanitized_content,
                message_for_system=None,
                user_language=None,
                error=f"{e}.\n{error_suffix}",
            )

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
