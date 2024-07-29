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


class ContextualRetrievalAgentResponse(AssistantResponse):
    pass