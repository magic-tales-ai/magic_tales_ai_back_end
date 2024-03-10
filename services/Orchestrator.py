from fastapi import WebSocket
from data_structures.command import Command
from data_structures.message import Message, MessageSchema, OriginEnum, TypeEnum
from data_structures.profile import Profile
from data_structures.profile import Profile, ProfileSchema
from data_structures.profile import ProfileSchema
from data_structures.story import Story
from data_structures.story import Story, StorySchema
from data_structures.user import User
from data_structures.user import User, UserSchema
from data_structures.user import UserSchema
from data_structures.ws_input import WSInput
from data_structures.ws_output import WSOutput
from services.SessionService import refresh_access_token
from sqlalchemy import desc, select
from sqlalchemy.orm import Session
from typing import List


class Orchestrator:
    def __init__(
        self, message_sender, chat_assistant, session: Session, websocket: WebSocket
    ):
        self.message_sender = message_sender
        self.chat_assistant = chat_assistant
        self.session = session
        self.websocket = websocket
        self.token_data = None
        self.user_id = None
        self.new_token = None

    async def process(self, request: WSInput, token_data: dict):
        print(request.token)  # DEBUG

        # Here actions for ALL COMMANDS to PROCESS

        # Refresh access token and set user_id if isn't try_mode
        if request.try_mode is False or None:
            self.token_data = token_data
            self.new_token = refresh_access_token(request.token)
            self.user_id = token_data.get("user_id")
        else:
            self.token_data = None
            self.new_token = None
            self.user_id = None

        # CONVERSATION FOR ALL COMMANDS
        conversation = Message(
            user_id=self.user_id,
            session_id=self.websocket.uid,
            command=request.command,
            origin=OriginEnum.user,
            type=(
                TypeEnum.chat
                if request.command == Command.USER_MESSAGE
                else TypeEnum.command
            ),
            details=request.model_dump(),
        )
        await self.add_message_to_session(conversation)

        # ORCH must process EVERY COMAND and delegate in CHAT ASSISTANT
        # ORCH send the ACK message to USER

        # NEW STORY
        if request.command == Command.NEW_TALE:
            # > send ACK to client
            await self.send_message_to_frontend(
                WSOutput(command=request.command, token=self.new_token, ack=True)
            )
            # > tell BOT to process the COMMMAND
            await self.chat_assistant.process(request)

        # NEW STORIE SPIN OFF (from existing storie)
        if request.command == Command.SPIN_OFF:
            if not request.story_id:
                raise Exception("story_id is required for spin-off")

            await self.send_message_to_frontend(
                WSOutput(
                    command=request.command,
                    token=self.new_token,
                    ack=True,
                    data={"story_parent_id": request.story_id},
                )
            )
            await self.chat_assistant.process(request)

        # UPDATE PROFILE
        if request.command == Command.USER_REQUEST_UPDATE_PROFILE:
            if not request.profile_id:
                raise Exception("profile_id is required for update profile")

            await self.send_message_to_frontend(
                WSOutput(command=request.command, token=self.new_token, ack=True)
            )
            await self.chat_assistant.process(request)

        # CONVERSATION RECOVERY
        # this command is necesary to recover a current and not finished conversation
        # the recover is based on CHAT TABLE records
        #
        if request.command == Command.CONVERSATION_RECOVERY:
            if not self.websocket.uid:
                raise Exception(
                    "uid is required for conversation recovery"
                )  # SESSION/CONVERSATION ID

            # read current conversation (CHAT ID)
            conversations = await self.get_conversations_by_session_id(
                self.websocket.uid
            )
            conversation_dicts = [
                MessageSchema().dump(conversation) for conversation in conversations
            ]
            # send full conversation to CLIENTE/USER (to rebuild it)
            await self.send_message_to_frontend(
                WSOutput(
                    command=request.command,
                    token=self.new_token,
                    data={"conversations": conversation_dicts},
                )
            )
            # > tell BOT to process the COMMMAND
            await self.chat_assistant.process(request)

        # ASSOCIATE USER WITH CONVERSATIONS
        if request.command == Command.LINK_USER_WITH_CONVERSATIONS:
            if not self.user_id:
                raise Exception("user_id is required for link user with conversations")

            if not request.session_ids:
                raise Exception(
                    "session_ids is required for link user with conversations"
                )  # SESSION/CONVERSATION ID

            await self.link_user_with_conversations(request.session_ids)
            await self.send_message_to_frontend(
                WSOutput(command=request.command, token=self.new_token, ack=True)
            )

        if request.command == Command.USER_MESSAGE:
            await self.chat_assistant.process(request)

        return 1

    # send_message_to_frontend()
    # method fr SENDING commands to CLIENT/USER from ORCHESTRATOR
    async def send_message_to_frontend(self, output_model: WSOutput):
        conversation = Message(
            user_id=self.user_id,
            session_id=self.websocket.uid,
            command=output_model.command,
            origin=OriginEnum.ai,
            type=TypeEnum.command,
            details=output_model.dict(),
        )
        await self.add_message_to_session(conversation)
        await self.message_sender.send_message_to_frontend(output_model)

    # ---------------------------------------------- Getters methods ----------------------------------------------

    async def get_user_by_id(self, id) -> User:
        return self.session.get(User, id)

    async def get_profile_by_id(self, id):
        return self.session.get(Profile, id)

    async def get_story_by_id(self, id):
        return self.session.get(Story, id)

    async def get_stories_by_user_id(self, user_id):
        profiles_ids = (
            self.session.execute(select(Profile.id).where(Profile.user_id == user_id))
            .scalars()
            .all()
        )
        return (
            self.session.execute(
                select(Story).filter(Story.profile_id.in_(profiles_ids))
            )
            .scalars()
            .all()
        )

    async def get_stories_by_profile_id(self, profile_id):
        return (
            self.session.execute(select(Story).where(Story.profile_id == profile_id))
            .scalars()
            .all()
        )

    async def get_conversations_by_session_id(self, session_id):
        return (
            self.session.execute(
                select(Message)
                .where(Message.session_id == session_id, Message.type == "chat")
                .order_by(desc(Message.session_id), Message.id)
            )
            .scalars()
            .all()
        )

    # ---------------------------------------------- End getters methods ------------------------------------------

    # ---------------------------------------------- Post methods -------------------------------------------------

    async def create_profile(self, data: Profile):
        instance = Profile(user_id=data.user_id, details=data.details)
        self.session.add(instance)
        self.session.commit()
        data.id = instance.id
        data.user = await self.get_user_by_id(data.user_id)

        return data

    async def create_story(self, data: Story):
        instance = Story(
            profile_id=data.profile_id,
            session_id=data.session_id,
            title=data.title,
            synopsis=data.synopsis,
            last_successful_step=data.last_successful_step,
        )
        self.session.add(instance)
        self.session.commit()
        data.id = instance.id

        return data

    async def add_message_to_session(self, data: Message):
        instance = Message(
            user_id=data.user_id,
            session_id=data.session_id,
            origin=data.origin,
            type=data.type,
            command=data.command,
            details=data.details,
        )
        self.session.add(instance)
        self.session.commit()
        data.id = instance.id
        # data.user = await self.get_user_by_id(data.user_id)

        return data

    async def delete_messages_by_session_id(self, session_id):  # (NOT USED)
        query = Message.__table__.delete().where(Message.session_id == session_id)
        self.session.execute(query)
        self.session.commit()

    # ---------------------------------------------- End post methods ---------------------------------------------

    # ---------------------------------------------- Update methods -----------------------------------------------

    async def update_profile_by_id(self, profile_id, data: Profile):
        profile = self.get_profile_by_id(profile_id)
        profile.details = data.details
        self.session.update(profile)
        self.session.commit()

        return profile

    async def link_user_with_conversations(self, session_ids):
        query = (
            Message.__table__.update()
            .where(Message.session_id.in_(session_ids))
            .values(user_id=self.user_id)
        )
        self.session.execute(query)
        self.session.commit()

    # ---------------------------------------------- End update methods -------------------------------------------
