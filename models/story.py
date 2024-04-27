import os
import markdown
import json
from typing import Any, Dict, List, Optional
import datetime
from db import Base
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship

from marshmallow import Schema, fields
from dataclasses import dataclass, field
import dill
import traceback

from models.profile import ProfileSchema
from models.story_state import StoryState

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)

STORY_DATA_FILENAME = "story.dill"


class Story(Base):
    __tablename__ = "stories"
    id = Column(Integer, primary_key=True)
    profile_id = Column(Integer, ForeignKey("profiles.id"))
    session_id = Column(String(255))
    story_folder = Column(Text)
    images_subfolder = Column(Text)
    title = Column(Text)
    features = Column(Text)
    synopsis = Column(Text)
    last_successful_step = Column(Integer)
    last_updated = Column(TIMESTAMP, default=datetime.datetime.now(datetime.UTC))
    created_at = Column(TIMESTAMP, default=datetime.datetime.now(datetime.UTC))
    # chapters = relationship("Chapter", back_populates="story")
    # images = relationship("Image", back_populates="story")

    def to_dict(self):
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "title": self.title,
            "features": self.features,
            "synopsis": self.synopsis,
            "last_successful_step": self.last_successful_step,
            "last_updated": self.last_updated,
            "created_at": self.created_at,
        }


# class Chapter(Base):
#     __tablename__ = "chapters"
#     id = Column(Integer, primary_key=True)
#     story_id = Column(Integer, ForeignKey("stories.id"))
#     chapter_number = Column(Integer)
#     content = Column(Text)
#     story = relationship("Story", back_populates="chapters")


# class Image(Base):
#     __tablename__ = "images"
#     id = Column(Integer, primary_key=True)
#     story_id = Column(Integer, ForeignKey("stories.id"))
#     image_description = Column(Text)
#     image_path = Column(Text)
#     story = relationship("Story", back_populates="images")


class StorySchema(Schema):
    id = fields.Int()
    profile_id = fields.Int()
    session_id = fields.Str()
    title = fields.Str()
    features = fields.Str()
    synopsis = fields.Str()
    last_successful_step = fields.Int()
    profile = fields.Nested(ProfileSchema, many=False)


@dataclass
class InMemStoryData:
    chapters: List[Dict[str, str]] = field(default_factory=list)
    post_processed_chapters: List[Dict[str, Any]] = field(default_factory=list)
    image_prompt_generator_prompt_messages: List[Dict[str, str]] = field(
        default_factory=list
    )
    image_prompts: List[Dict[str, Any]] = field(default_factory=list)
    image_filenames: List[str] = field(default_factory=list)

    def save_state(
        self, story_folder: str, filename: str = STORY_DATA_FILENAME
    ) -> None:
        """Save the state of the InMemStoryData instance to a file."""
        filepath = os.path.join(story_folder, filename)
        try:
            with open(filepath, "wb") as f:
                dill.dump(self, f)
            logger.info(f"Successfully saved InMemStoryData state to {filepath}")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(
                f"Could not save InMemStoryData state to {filepath}.\nException: {e}\n{tb}"
            )

    @staticmethod
    def load_state(
        story_folder: str, filename: str = STORY_DATA_FILENAME
    ) -> "InMemStoryData":
        """Load the state of a InMemStoryData instance from a file."""
        filepath = os.path.join(story_folder, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    loaded_data = dill.load(f)
                logger.info(f"Successfully loaded InMemStoryData state from {filepath}")
                return loaded_data
            except Exception as e:
                logger.error(f"Could not load InMemStoryData state from {filepath}. Error: {e}", exc_info=True)
        
        logger.info("Returning a new instance of InMemStoryData.")
        return InMemStoryData()  # Return a new instance if the load fails



def handle_custom_objects(obj):
    if isinstance(obj, StoryState):
        return obj.name
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")
