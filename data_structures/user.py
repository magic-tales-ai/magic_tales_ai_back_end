from db import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP
from marshmallow import Schema, fields


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP)


class UserSchema(Schema):
    id = fields.Int()
    email = fields.Str()
