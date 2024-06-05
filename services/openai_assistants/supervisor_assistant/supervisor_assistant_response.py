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


class SupervisorAssistantResponse(AssistantResponse):
    def __init__(
        self,
        message_for_user: Optional[str] = None,
        intervention_needed: Optional[bool] = None,
        error: Optional[str] = None,
    ):
        super().__init__(message_for_user=message_for_user, error=error)
        self.intervention_needed = intervention_needed

    async def serialize(self):
        return {
            "message_for_user": self.message_for_user,
            "intervention_needed": self.intervention_needed,
            "error": self.error,
        }

    async def get_intervention_needed(self) -> Optional[bool]:
        return self.intervention_needed
