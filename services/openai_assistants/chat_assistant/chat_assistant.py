from typing import Any
import traceback
import os
import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Callable
from openai import AsyncOpenAI
from openai.types.beta.threads import (
    ThreadMessage,
    MessageContentText,
    MessageContentImageFile,
)

from services.utils.log_utils import get_logger
from services.openai_assistants.assistant import Assistant
from models.ws_input import WSInput
from models.user import User

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
        self.chat_completed_event = asyncio.Event()

    async def wait_for_chat_completion(self):
        """
        Waits for the chat to complete. This method blocks until the chat_complete_event is set.
        """
        await self.chat_completed_event.wait()

    def _extract_with_fallbacks(
        self, ai_message_content: str
    ) -> Tuple[Optional[str], Optional[dict]]:
        """
        Attempts to directly parse AI message content as JSON and extract `message_for_human` and `message_for_system`.

        Args:
            ai_message_content (str): The raw AI response content.

        Returns:
            A tuple containing potentially extracted `message_for_human` as str and
            `message_for_system` as a dict or None for each if not applicable or errors occur.
        """
        try:
            # Directly parse the AI message content as JSON
            ai_message_dict = json.loads(ai_message_content)
            message_for_human = ai_message_dict.get("message_for_human")
            message_for_system = ai_message_dict.get(
                "message_for_system"
            )  # This will be a dict or None

            return message_for_human, message_for_system

        except json.JSONDecodeError as e:
            try:
                message_for_human = re.search(
                    r'"message_for_human":\s*"([^"]*)"', ai_message_content
                ).group(1)
                message_for_system = re.search(
                    r'"message_for_system":\s*"([^"]*)"', ai_message_content
                ).group(1)
                return message_for_human, message_for_system
            except Exception as e:
                logger.error(
                    f"Failed to parse AI message content: {ai_message_content}/n/nError traceback:/n/n {traceback.format_exc()}"
                )
                retry_message = " I apologize, I encountered a hiccup processing your message. Could you rephrase or try again?"
                message_for_human = (message_for_human or "") + retry_message
                return message_for_human, None

    def _default_parsing(
        self, ai_message_content: str
    ) -> Tuple[Optional[str], Optional[WSInput]]:
        """
        Robustly parses AI response content, extracting `message_for_human` and creating a WSInput instance for `message_for_system`.
        """
        message_for_human, message_for_system_pre_processed = (
            self._extract_with_fallbacks(ai_message_content)
        )

        message_for_system = None
        if message_for_system_pre_processed:
            try:
                # Check if message_for_system_str is a string and attempt to parse it into a dict
                if isinstance(message_for_system_pre_processed, str):
                    message_for_system_dict = json.loads(
                        message_for_system_pre_processed
                    )
                elif isinstance(message_for_system_pre_processed, dict):
                    message_for_system_dict = message_for_system_pre_processed
                else:
                    raise ValueError(
                        "message_for_system is neither a dict nor a stringifiable JSON."
                    )

                # Now create WSInput instance from the properly parsed dict
                message_for_system = WSInput(**message_for_system_dict)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"Error creating WSInput from system message: {e}")
                # User-friendly message in case of failure
                retry_message = " I apologize, I encountered a hiccup processing a system request. Could you rephrase or try again?"
                message_for_human = (message_for_human or "") + retry_message

        return message_for_human, message_for_system
