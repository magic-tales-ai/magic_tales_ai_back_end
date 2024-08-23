from typing import Union, Optional, Literal
from openai.types.beta.assistant_response_format_option_param import (
    ResponseFormatText,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
)

import logging

from services.openai_assistants.assistant_response.assistant_response import (
    AssistantResponse,
)
from services.utils.log_utils import get_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
# Get a logger instance for this module
logger = get_logger(__name__)


class Command(ResponseFormatJSONObject):
    """Base class for all commands."""

    command: str


class UpdateProfile(Command):
    """Command to update an existing profile."""

    profile_id: int
    current_name: str
    current_age: str
    user_id: str
    updated_name: Optional[str] = None
    updated_age: Optional[str] = None
    updated_details: Optional[str] = None


class NewProfile(Command):
    """Command to create a new profile."""

    name: str
    age: int
    details: str


class ContinueUnfinishedStory(Command):
    """Command to continue an unfinished story."""

    continue_where_we_left_off: bool


class StartStoryGeneration(Command):
    """Command to start a new story generation process."""

    name: str
    age: int
    user_id: str


class StructuredResponse(ResponseFormatJSONObject):
    """Response model containing the message for the user, the system command, and the user language."""

    message_for_user: str
    message_for_system: Optional[
        Union[UpdateProfile, NewProfile, ContinueUnfinishedStory, StartStoryGeneration]
    ] = None
    user_language: str


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
        logger.info(f"Returning user_language: {self.user_language}")
        return self.user_language

    async def get_message_for_system(self) -> Optional[dict]:
        return self.message_for_system
