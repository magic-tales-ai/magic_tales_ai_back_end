import datetime
import enum
from db import Base
from marshmallow import Schema, fields
from data_structures.user import UserSchema
from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, Enum, JSON


class OriginEnum(enum.Enum):
    user = "user"
    ai = "ai"


class TypeEnum(enum.Enum):
    chat = "chat"
    command = "command"


class Message(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    user_id = Column(ForeignKey("users.id"))
    session_id = Column(String(255))
    origin = Column(Enum(OriginEnum))
    type = Column(Enum(TypeEnum))
    command = Column(String(255))
    details = Column(JSON)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)


class MessageSchema(Schema):
    id = fields.Int()
    user_id = fields.Int()
    user = fields.Nested(UserSchema, many=False)
    session_id = fields.Str()
    origin = fields.Enum(OriginEnum)
    type = fields.Enum(TypeEnum)
    command = fields.Str()
    details = fields.Dict()
