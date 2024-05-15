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
        message_for_human = None
        message_for_system = {}

        try:
            # Remove any surrounding backticks and newline characters
            ai_message_content = ai_message_content.strip("```\n")

            # Directly parse the AI message content as JSON
            ai_message_dict = json.loads(ai_message_content)
            message_for_human = ai_message_dict.get("message_for_human")
            message_for_system = ai_message_dict.get("message_for_system", {})
            if not isinstance(message_for_system, dict):
                message_for_system = {}
            return message_for_human, message_for_system, None
        except json.JSONDecodeError as e:
            human_message_pattern = r'"message_for_human"\s*:\s*"([^"]*)"'
            system_message_pattern = r'"message_for_system"\s*:\s*"({[^"]*})"'

            try:
                human_match = re.search(human_message_pattern, ai_message_content)
                if human_match:
                    message_for_human = human_match.group(1)

                system_match = re.search(system_message_pattern, ai_message_content, re.DOTALL)
                if system_match:
                    message_for_system_str = system_match.group(1)
                    try:
                        message_for_system = json.loads(message_for_system_str)
                    except json.JSONDecodeError:
                        # If the extracted message_for_system is not valid JSON, try to parse it as a string
                        message_for_system = {"command": message_for_system_str}
                else:
                    message_for_system = {}

                return message_for_human, message_for_system, None
            except (KeyError, ValueError) as e:
                logger.error(f"Failed to parse AI message content: {ai_message_content}\n\nError: {str(e)}")
                return None, {}, traceback.format_exc()

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
