import datetime
from db import Base
from sqlalchemy import Column, Integer, Text, TIMESTAMP
from sqlalchemy.sql.schema import ForeignKey
from sqlalchemy.orm import relationship
from marshmallow import Schema, fields
from marshmallow.fields import Nested

from models.user import UserSchema


class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True)
    details = Column(Text)
    user_id = Column(ForeignKey("users.id"))
    user = relationship("User", lazy="joined")
    created_at = Column(TIMESTAMP, default=datetime.datetime.now(datetime.UTC))

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "details": self.details,
            "created_at": self.created_at,
        }


class ProfileSchema(Schema):
    id = fields.Int()
    details = fields.Str()
    user_id = fields.Int()
    user = fields.Nested(UserSchema, many=False)
