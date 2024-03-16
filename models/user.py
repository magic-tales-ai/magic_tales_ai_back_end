from db import Base
from sqlalchemy import Column, Integer, String, TIMESTAMP
from marshmallow import Schema, fields


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    assistant_id = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP)

    def __str__(self):
        return f"User info: username='{self.username}', email='{self.email}', created_at='{self.created_at}')"
    
    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at
        }    

class UserSchema(Schema):
    id = fields.Int()
    email = fields.Str()
