from typing import List
from data_structures.Message import Message, OriginEnum, TypeEnum
from data_structures.profile import Profile
from data_structures.story import Story
from data_structures.story_state import StoryState
from data_structures.ws_input import WSInput
from data_structures.ws_output import WSOutput
from data_structures.command import Command
from services.MessageSender import MessageSender
from services.Orchestrator import Orchestrator

from services.SessionService import refresh_access_token

####
import traceback
import os
import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Tuple, Callable, Any
from openai import AsyncOpenAI
from services.utils.log_utils import get_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class ChatAssistant:
    def __init__(
        self, config, message_sender: MessageSender, orchestrator: Orchestrator
    ):
        """
        Initialize the Chat Assistant.

        Args:
            config (DictConfig): Configuration parameters.
            message_sender (MessageSender): Class that handles the websocket communication between the Assistant, the Orchestrator and the Front-End
            orchestrator (Orchestrator): Core orchestrator for Magic Tales application. Manages the story generation process.
        """
        self._validate_openai_api_key()
        self.config = config
        self.client = AsyncOpenAI()
        self.user_facing_assistant = None
        self.user_facing_thread = None
        self.user_facing_chat_info = []
        self.chat_completed_event = asyncio.Event()

        # message_sender: object to send messages to CLIENT/USER
        self.message_sender = message_sender
        # orchestrator
        self.orchestrator = orchestrator

        # command_messages: OBJECT USED FOR FAKE RESPONSES
        self.command_messages = {}

        self.new_token = None  # prop for AUTO RENEW JWT TOKENS (auth API tokens)

        self.user_id = None  # ID logged USER

        # VAR/PROPS for FAKE
        # current working profile, current working tale and current working spin-off tale
        # this props don't do nothing on the FAKE version, just contains the data of current process
        # but this is something the REAL BOT will do on his way.
        # spin_off is not necesary too, is just another TALE
        self.profile = None
        self.tale = None
        self.spin_off = None

    async def process(self, request: WSInput):
        # NEW-TALE
        if request.command == Command.NEW_TALE:
            output_model = WSOutput(
                command=Command.MESSAGE_FOR_HUMAN,
                token=self.orchestrator.new_token,
                message="Hi!, what would you like to write about?",
            )
            await self.send_message_to_frontend(output_model)  # FIRST MESSAGE RECEIVED!
            self.command_messages[Command.NEW_TALE] = 1  # FIRST FAKE RESPONSE
            user = await self.orchestrator.get_user_by_id(
                self.orchestrator.user_id
            )  # USEFUL DATA

        # SPIN-OFF
        if request.command == Command.SPIN_OFF:
            output_model = WSOutput(
                command=Command.MESSAGE_FOR_HUMAN,
                token=self.orchestrator.new_token,
                message="How would you like to continue your story?",
            )
            await self.send_message_to_frontend(output_model)  # FIRST MESSAGE RECEIVED!
            self.command_messages[Command.SPIN_OFF] = 1  # FIRST FAKE RESPONSE
            self.tale = await self.orchestrator.get_story_by_id(
                request.story_id
            )  # USEFUL DATA

        # USER-REQ-UPDATE-PROFILE
        if request.command == Command.USER_REQUEST_UPDATE_PROFILE:
            output_model = WSOutput(
                command=Command.MESSAGE_FOR_HUMAN,
                token=self.orchestrator.new_token,
                message="Hi!, which user would you like to update?",
            )
            await self.send_message_to_frontend(output_model)  # FIRST MESSAGE RECEIVED!
            self.command_messages[Command.USER_REQUEST_UPDATE_PROFILE] = (
                1  # FIRST FAKE RESPONSE
            )
            profile = self.orchestrator.get_profile_by_id(
                request.profile_id
            )  # USEFUL DATA

        # CONVERSATION RECOVERY
        if request.command == Command.CONVERSATION_RECOVERY:
            self.command_messages[Command.CONVERSATION_RECOVERY] = (
                1  # FIRST FAKE RESPONSE
            )

        # USER MESSAGES
        if request.command == Command.USER_MESSAGE:

            # NEW-TALE MESSAGE
            if Command.NEW_TALE in self.command_messages:

                # IF IT'S USER FIRST MESSAGE WE CONTINUE THE CONVERSATION, UPDATE STATUS AND SEND PROGRESS
                if self.command_messages[Command.NEW_TALE] == 1:
                    self.command_messages[Command.NEW_TALE] = 2
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.MESSAGE_FOR_HUMAN,
                            token=self.orchestrator.new_token,
                            message="That's a great story! Tell me some more little details about this.",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.STATUS_UPDATE,
                            token=self.orchestrator.new_token,
                            message="Writing the most epic story",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.PROGRESS_UPDATE,
                            token=self.orchestrator.new_token,
                            progress_percent=20,
                            story_state=StoryState.IMAGE_PROMPT_GENERATION,
                            message="You've reached {perc} completion in building your story. Keep it going!",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.MESSAGE_FOR_HUMAN,
                            token=self.orchestrator.new_token,
                            images=[
                                "http://104.237.150.104:8001/static/images/img-6.jpg",
                                "http://104.237.150.104:8001/static/images/img-7.jpg",
                                "http://104.237.150.104:8001/static/images/img-4.jpg",
                            ],
                        )
                    )

                # IF IT'S USER SECOND MESSAGE WE END THE CONVERSATION, UPDATE STATUS, SEND PROGRESS AND NOTIFY ABOUT THE PROCESS COMPLETATION
                elif self.command_messages[Command.NEW_TALE] == 2:
                    self.command_messages[Command.NEW_TALE] = 3
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.MESSAGE_FOR_HUMAN,
                            token=self.orchestrator.new_token,
                            message="Congratulations! I think we're making something good here.",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.STATUS_UPDATE,
                            token=self.orchestrator.new_token,
                            message="Finishing the last details...",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.PROGRESS_UPDATE,
                            token=self.orchestrator.new_token,
                            progress_percent=100,
                            story_state=StoryState.FINAL_DOCUMENT_GENERATED,
                            message="You've achieved {perc} completion in crafting your narrative.",
                        )
                    )
                    profile = await self.orchestrator.create_profile(
                        Profile(
                            user_id=self.orchestrator.user_id,
                            details='{"name": "Carlos", "last_name": "Smith", "age": 8}',
                        )
                    )
                    self.profile = profile
                    self.tale = await self.orchestrator.create_story(
                        Story(
                            profile_id=profile.id,
                            session_id=self.orchestrator.websocket.uid,
                            title="test",
                            synopsis="test",
                            last_successful_step=2,
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.CHAT_COMPLETED,
                            token=self.orchestrator.new_token,
                            files=[
                                "http://104.237.150.104:8001/static/files/Sample.pdf"
                            ],
                            data={"story_id": self.tale.id},
                        )
                    )

            # SPIN-OFF MESSAGE
            if Command.SPIN_OFF in self.command_messages:

                # IF IT'S USER FIRST MESSAGE WE CONTINUE THE CONVERSATION UPDATE STATUS AND SEND PROGRESS
                if self.command_messages[Command.SPIN_OFF] == 1:
                    self.command_messages[Command.SPIN_OFF] = 2
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.MESSAGE_FOR_HUMAN,
                            token=self.orchestrator.new_token,
                            message="Too interesting... Would you like to add anything else?",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.STATUS_UPDATE,
                            token=self.orchestrator.new_token,
                            message="Writing the most epic spin-off",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.PROGRESS_UPDATE,
                            token=self.orchestrator.new_token,
                            progress_percent=60,
                            story_state=StoryState.DOCUMENT_GENERATION,
                            message="Great progress! You're right at the {perc} mark.",
                        )
                    )

                # IF IT'S USER SECOND MESSAGE WE END THE CONVERSATION, UPDATE STATUS, SEND PROGRESS AND NOTIFY ABOUT THE PROCESS COMPLETATION
                elif self.command_messages[Command.SPIN_OFF] == 2:
                    self.command_messages[Command.SPIN_OFF] = 3
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.MESSAGE_FOR_HUMAN,
                            token=self.orchestrator.new_token,
                            message="Amazing!, it'll be a nice spin-off.",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.STATUS_UPDATE,
                            token=self.orchestrator.new_token,
                            message="Finishing the last details...",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.PROGRESS_UPDATE,
                            token=self.orchestrator.new_token,
                            progress_percent=90,
                            story_state=StoryState.USER_FACING_CHAT,
                            message="You're about to finish! You're right at the {perc} mark.",
                        )
                    )
                    self.spin_off = await self.orchestrator.create_story(
                        Story(
                            profile_id=self.tale.profile_id,
                            session_id=self.orchestrator.websocket.uid,
                            title="spin-off test",
                            synopsis="spin-off test",
                            last_successful_step=2,
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.CHAT_COMPLETED,
                            token=self.orchestrator.new_token,
                            files=[
                                "http://104.237.150.104:8001/static/files/Sample.pdf"
                            ],
                            data={"story_id": self.spin_off.id},
                        )
                    )

            # USER-REQ-UPDATE-PROFILE MESSAGE
            if Command.USER_REQUEST_UPDATE_PROFILE in self.command_messages:

                # IF IT'S USER FIRST MESSAGE WE CONTINUE THE CONVERSATION AND SEND PROGRESS
                if self.command_messages[Command.USER_REQUEST_UPDATE_PROFILE] == 1:
                    self.command_messages[Command.USER_REQUEST_UPDATE_PROFILE] = 2
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.MESSAGE_FOR_HUMAN,
                            token=self.orchestrator.new_token,
                            message="That's great profile. Anything else?",
                        )
                    )

                # IF IT'S USER SECOND MESSAGE WE END THE CONVERSATION, SEND PROGRESS AND NOTIFY ABOUT THE PROCESS COMPLETATION
                elif self.command_messages[Command.USER_REQUEST_UPDATE_PROFILE] == 2:
                    self.command_messages[Command.USER_REQUEST_UPDATE_PROFILE] = 3
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.MESSAGE_FOR_HUMAN,
                            token=self.orchestrator.new_token,
                            message="Great. We're done.",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.CHAT_COMPLETED,
                            token=self.orchestrator.new_token,
                        )
                    )

            # CONVERSATION RECOVERY MESSAGE
            if Command.CONVERSATION_RECOVERY in self.command_messages:

                # NOTES TO REAL BOT:
                # Here, the chat bot needs to READ the FULL CONTEXT of the conversation
                # to put it self in the same state it was.
                # can get the CONTEXT from its own (something managed by the BOT) or reading the DATABASE (chat history)

                # IF THE CONVERSATION IS RECOVERED WE CONTINUE THE CONVERSATION AND NOTIFY ABOUT THE PROCESS COMPLETATION
                if self.command_messages[Command.CONVERSATION_RECOVERY] == 1:
                    self.command_messages[Command.CONVERSATION_RECOVERY] = 2
                    profile = await self.orchestrator.create_profile(
                        Profile(
                            user_id=self.orchestrator.user_id,
                            details='{"name": "Carlos", "last_name": "Smith", "age": 8}',
                        )
                    )
                    self.profile = profile
                    self.tale = await self.orchestrator.create_story(
                        Story(
                            profile_id=profile.id,
                            session_id=self.orchestrator.websocket.uid,
                            title="test",
                            synopsis="test",
                            last_successful_step=2,
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.MESSAGE_FOR_HUMAN,
                            token=self.orchestrator.new_token,
                            message="Great. We're done.",
                        )
                    )
                    await self.send_message_to_frontend(
                        WSOutput(
                            command=Command.CHAT_COMPLETED,
                            token=self.orchestrator.new_token,
                            files=[
                                "http://104.237.150.104:8001/static/files/Sample.pdf"
                            ],
                            data={"story_id": self.tale.id},
                        )
                    )

    # send_message_to_frontend()
    # method fr SENDING commands to CLIENT/USER from CHAT BOT
    async def send_message_to_frontend(self, output_model: WSOutput):
        conversation = Message(
            user_id=self.orchestrator.user_id,
            session_id=self.orchestrator.websocket.uid,
            command=output_model.command,
            origin=OriginEnum.ai,
            type=TypeEnum.chat,
            details=output_model.dict(),
        )
        await self.orchestrator.add_message_to_session(conversation)
        await self.message_sender.send_message_to_frontend(output_model)
