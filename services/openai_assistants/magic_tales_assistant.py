from typing import TypeVar, Generic

from services.openai_assistants.assistant import Assistant
from services.openai_assistants.persona_rag.agents.agent import Agent as PersonaRAGAgent
from services.utils.log_utils import get_logger
from services.openai_assistants.assistant_response.assistant_response import (
    AssistantResponse,
)
from services.openai_assistants.assistant_input.assistant_input import (
    AssistantInput,
    Source,
)

TInput = TypeVar("TInput", bound=AssistantInput)
TResponse = TypeVar("TResponse", bound=AssistantResponse)


class MagicTalesAgent(Assistant, PersonaRAGAgent, Generic[TInput, TResponse]):
    def __init__(self, config):
        Assistant.__init__(self, config)
        PersonaRAGAgent.__init__(self, template=None, model=config.model, key_map=None)
        
    def _default_parsing(self, ai_message_content: str):
        # Implement default parsing logic
        pass

    def _default_error_processing_request(self, message: str, error: str):
        # Implement default error processing logic
        pass
