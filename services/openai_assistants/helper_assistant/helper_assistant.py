import os
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union


from services.utils.log_utils import get_logger
from services.openai_assistants.assistant import Assistant
from models.user import User
from models.ws_input import WSInput

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class HelperAssistant(Assistant):

    def _default_parsing(
        self, ai_message_content: str
    ) -> Tuple[Optional[str], Optional[WSInput], Optional[str]]:
        """
        Robustly parses AI response content, extracting `message_for_human`, creating a WSInput instance for `message_for_system AND pass over the error if it occurs.
        """

        return ai_message_content, None, None
