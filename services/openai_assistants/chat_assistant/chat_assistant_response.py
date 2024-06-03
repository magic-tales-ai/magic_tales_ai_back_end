from typing import Optional
import logging

from services.openai_assistants.assistant_response.assistant_response import (
    AssistantResponse,
)
from services.utils.log_utils import get_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
# Get a logger instance for this module
logger = get_logger(__name__)


class ChatAssistantResponse(AssistantResponse):
    def __init__(
        self,
        message_for_user: Optional[str] = None,
        message_for_system: Optional[dict] = None,
        user_language: Optional[str] = None,
        error: Optional[str] = None,
    ):
        super().__init__(message_for_user=message_for_user, error=error)
        self.message_for_system = message_for_system
        self.user_language = user_language

    async def serialize(self):
        return {
            "message_for_user": self.message_for_user,
            "message_for_system": self.message_for_system,
            "user_language": self.user_language,
            "error": self.error,
        }

    async def get_user_language(self) -> Optional[str]:
        logger.info(
            f"Returning user_language: {self.user_language}"
        )  # Add logging here
        return self.user_language

    async def get_message_for_system(self) -> Optional[dict]:
        return self.message_for_system
