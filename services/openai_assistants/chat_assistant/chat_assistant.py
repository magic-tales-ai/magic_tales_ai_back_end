from typing import Any, Dict, List, Optional, Tuple
import traceback
import asyncio
import json
import logging
import re


from services.utils.log_utils import get_logger
from services.openai_assistants.assistant import Assistant
from magic_tales_models.models.ws_input import WSInput
from magic_tales_models.models.user import User

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class ChatAssistant(Assistant):
    def __init__(self, config):
        """
        Initialize the Chat Assistant.

        Args:
            config (DictConfig): Configuration parameters.
        """
        super().__init__(config)        
    
    def _parse_with_fallbacks(
        self, ai_message_content: str
    ) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
        """
        Attempts to directly parse AI message content as JSON and extract `message_for_human` and `message_for_system`.

        Args:
            ai_message_content (str): The raw AI response content.

        Returns:
            A tuple containing potentially extracted `message_for_human` as str and `message_for_system` as a dict or None for each if not applicable or errors occur.
        """
        try:
            # # Remove surrounding characters and escape single quotes
            # sanitized_content = ai_message_content.strip("\`\n").replace("'", "\\'")

            # # Remove leading and trailing quotes
            # sanitized_content = sanitized_content.strip('"')

            # # Replace escaped double quotes with single quotes
            # sanitized_content = sanitized_content.replace('\\"', '"')

            sanitized_content =  ai_message_content

            # Parse the sanitized JSON
            ai_message_dict = json.loads(sanitized_content)

            # Extract the required keys
            message_for_human = ai_message_dict.get("message_for_human", "")
            message_for_system = ai_message_dict.get("message_for_system", {})

            # Ensure message_for_system is a dictionary
            if not isinstance(message_for_system, dict):
                logger.warning("message_for_system is not a dictionary. Resetting to empty dict.")
                message_for_system = {}

            return message_for_human, message_for_system, None

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"JSON parsing failed, attempting regex extraction: {str(e)}")
            return self._extract_with_regex(ai_message_content)
        

    def _extract_with_regex(self, ai_message_content: str) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
        """
        Fallback regex extraction if JSON parsing fails.

        Args:
            ai_message_content (str): The raw AI response content that failed JSON parsing.

        Returns:
            Tuple containing message for human, system command as a dictionary or None, and error message if applicable.
        """
        try:
            # Remove surrounding characters and escape single quotes
            sanitized_content = ai_message_content.strip("\`\n").replace("'", "\\'")

            # Extract message_for_human
            human_match = re.search(r'"message_for_human"\s*:\s*"([^"]*)"', sanitized_content)
            message_for_human = human_match.group(1) if human_match else ""

            # Extract message_for_system
            system_match = re.search(r'"message_for_system"\s*:\s*(\{.*?\})', sanitized_content, re.DOTALL)
            system_msg_str = system_match.group(1) if system_match else "{}"
            message_for_system = json.loads(system_msg_str)

            return message_for_human, message_for_system, None

        except Exception as e:
            error_msg = f"Failed during regex extraction: {traceback.format_exc()}"
            logger.warning(error_msg)
            return "", {}, error_msg
        
    def _default_parsing(
        self, ai_message_content: str
    ) -> Tuple[Optional[str], Optional[WSInput], Optional[str]]:
        """
        Robustly parses AI response content, extracting `message_for_human`, creating a WSInput instance for `message_for_system AND pass over the error if it occurs.
        """
        message_for_human, message_for_system_pre_processed, error = (
            self._parse_with_fallbacks(ai_message_content)
        )
        if error:
            return message_for_human, message_for_system_pre_processed, error

        message_for_system = {}
        if message_for_system_pre_processed:
            try:
                # create WSInput instance from the properly parsed dict
                message_for_system = WSInput(**message_for_system_pre_processed)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"Error creating WSInput from system message: {e}")
                error = traceback.format_exc()

        return message_for_human, message_for_system, error
