from services.openai_assistants.assistant_response.assistant_response import (
    AssistantResponse,
)

from pydantic import BaseModel


class StructuredResponse(BaseModel):
    """Response model"""

    pass


class HelperAssistantResponse(AssistantResponse):
    pass
