import enum
from marshmallow import Schema, fields
from models.user import UserSchema


class OriginEnum(enum.Enum):
    user = "user"
    ai = "ai"


class TypeEnum(enum.Enum):
    chat = "chat"
    command = "command"


class MessageSchema(Schema):
    id = fields.Int()
    user_id = fields.Int()
    user = fields.Nested(UserSchema, many=False)
    ws_session_uid = fields.Str()
    origin = fields.Enum(OriginEnum)
    type = fields.Enum(TypeEnum)
    command = fields.Str()
    details = fields.Dict()
