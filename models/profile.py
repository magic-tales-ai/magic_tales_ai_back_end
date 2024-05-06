from sqlalchemy.orm import relationship
from marshmallow import Schema, fields

from models.user import UserSchema

class ProfileSchema(Schema):
    id = fields.Int()
    details = fields.Str()
    user_id = fields.Int()
    user = fields.Nested(UserSchema, many=False)
