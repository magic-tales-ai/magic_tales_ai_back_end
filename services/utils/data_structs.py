import os
import json
import markdown
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import dill
import traceback

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class Command:
    # COMMANDS from USER to CORE-AI
    NEW_TALE = "new_tale"
    SPIN_OFF = "spin_off"
    UPDATE_PROFILE = "user_req_update_profile"
    CONVERSATION_RECOVERY = "conversation_recovery"
    LINK_USER_WITH_CONVERSATIONS = "link_user_with_conversations"
    USER_MESSAGE = "user_message"

    # COMMANDS from ORCH to USER
    # ACKNOWLEDGEMENT: REPEAT USER COMMAND and add ACK flag
    # ACKNOWLEDGEMENT = "ack" (not a real command)

    # COMMANDS from CHAT to USER
    PROGRESS_UPDATE = "progress_update"
    MESSAGE_FOR_HUMAN = "message_for_human"
    CHAT_COMPLETED = "done"
    STATUS_UPDATE = "status_update"


class StoryState:
    USER_FACING_CHAT = "USER_FACING_CHAT"
    STORY_TITLE_GENERATION = "STORY_TITLE_GENERATION"
    STORY_GENERATION = "STORY_GENERATION"
    IMAGE_PROMPT_GENERATION = "IMAGE_PROMPT_GENERATION"
    IMAGE_GENERATION = "IMAGE_GENERATION"
    DOCUMENT_GENERATION = "DOCUMENT_GENERATION"
    FINAL_DOCUMENT_GENERATED = "FINAL_DOCUMENT_GENERATED"


class ResponseStatus:
    DONE = "DONE"
    STARTED = "STARTED"
    FAILED = "FAILED"
