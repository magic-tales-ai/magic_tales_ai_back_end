import os
import markdown
import json
from typing import Any, Dict, List, Optional
import datetime
from db import Base
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
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
    profile_id = Column(Integer)
    session_id = Column(String(255))
    title = Column(Text)
    features = Column(Text)
    synopsis = Column(Text)
    last_successful_step = Column(Integer)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "title": self.title,
            "features": self.features,
            "synopsis": self.synopsis,
            "last_successful_step": self.last_successful_step,
            "created_at": self.created_at,
        }


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
class StoryData:
    story_folder: str = ""
    images_subfolder: str = "images"
    personality_profile: str = ""
    story_features: str = ""
    synopsis: str = ""
    title: Optional[str] = None
    chapters: List[Dict[str, str]] = field(default_factory=list)
    post_processed_chapters: List[Dict[str, Any]] = field(default_factory=list)
    image_prompt_generator_prompt_messages: List[Dict[str, str]] = field(
        default_factory=list
    )
    image_prompts: List[Dict[str, Any]] = field(default_factory=list)
    image_filenames: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "step": StoryState.USER_FACING_CHAT,
            "last_modified": None,
        }
    )

    def export_to_human_readable_format(self) -> None:
        """
        Export the content of the StoryData instance into human-readable files.

        Args:
            None

        Returns:
            None
        """
        # Create the main story folder
        story_folder_path = os.path.join(self.story_folder, "StoryData")
        os.makedirs(story_folder_path, exist_ok=True)

        # Personality Profile
        with open(os.path.join(story_folder_path, "Personality_Profile.md"), "w") as f:
            f.write(markdown.markdown(self.personality_profile))

        # Story Features
        with open(os.path.join(story_folder_path, "Story_Features.md"), "w") as f:
            f.write(markdown.markdown(self.story_features))

        # Synopsis
        with open(os.path.join(story_folder_path, "Synopsis.md"), "w") as f:
            f.write(markdown.markdown(self.synopsis))

        # Title
        if self.title:
            with open(os.path.join(story_folder_path, "Title.txt"), "w") as f:
                f.write(self.title)

        # Image Prompt Generator Messages
        md = "## Image Prompt Generator Messages\n"
        for idx, message in enumerate(self.image_prompt_generator_prompt_messages):
            md += f"\n### Prompt {idx + 1}\n"
            md += "\n#### System Message:\n"
            md += (
                "-"
                if message["system_message"] == ""
                else message["system_message"].replace("\n", "  \n")
            )
            md += "\n#### Human Message:\n"
            md += (
                "-"
                if message["human_message"] == ""
                else message["human_message"].replace("\n", "  \n")
            )

        with open(
            os.path.join(story_folder_path, "Image_Prompt_Generator_Messages.md"), "w"
        ) as f:
            f.write(md)

        # Chapters
        chapters_folder = os.path.join(story_folder_path, "Chapters")
        os.makedirs(chapters_folder, exist_ok=True)
        for idx, chapter in enumerate(self.chapters):
            with open(os.path.join(chapters_folder, f"Chapter_{idx + 1}.md"), "w") as f:
                f.write(markdown.markdown(chapter["content"]))

        # Post-Processed Chapters
        post_processed_folder = os.path.join(
            story_folder_path, "Post_Processed_Chapters"
        )
        os.makedirs(post_processed_folder, exist_ok=True)
        for idx, chapter in enumerate(self.post_processed_chapters):
            with open(
                os.path.join(
                    post_processed_folder, f"Post_Processed_Chapter_{idx + 1}.json"
                ),
                "w",
            ) as f:
                json.dump(chapter, f, indent=4)

        # Image Prompts
        image_prompts_folder = os.path.join(story_folder_path, self.images_subfolder)
        os.makedirs(image_prompts_folder, exist_ok=True)
        for idx, image_prompt in enumerate(self.image_prompts):
            with open(
                os.path.join(image_prompts_folder, f"Image_Prompt_{idx + 1}.json"), "w"
            ) as f:
                json.dump(image_prompt, f, indent=4)

        # Image Filenames
        with open(os.path.join(story_folder_path, "Image_Filenames.txt"), "w") as f:
            for filename in self.image_filenames:
                f.write(f"{filename}\n")

        # Metadata
        with open(os.path.join(story_folder_path, "Metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4, default=handle_custom_objects)

    def save_state(self, filename: str = STORY_DATA_FILENAME) -> None:
        """Save the state of the StoryData instance to a file."""
        filepath = os.path.join(self.story_folder, filename)
        try:
            with open(filepath, "wb") as f:
                dill.dump(self, f)
            logger.info(f"Successfully saved StoryData state to {filepath}")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(
                f"Could not save StoryData state to {filepath}.\nException: {e}\n{tb}"
            )

    def load_state(
        directory: str, filename: str = STORY_DATA_FILENAME
    ) -> Optional["StoryData"]:
        """Load the state of a StoryData instance from a file."""
        filepath = os.path.join(directory, filename)

        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    loaded_data = dill.load(f)
                logger.info(f"Successfully loaded StoryData state from {filepath}")
                return loaded_data
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(
                    f"Could not load StoryData state from {filepath}.\nException: {e}\n{tb}"
                )
                return None
        else:
            logger.error(f"File {filepath} does not exist.")
            return None


def handle_custom_objects(obj):
    if isinstance(obj, StoryState):
        return obj.name
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")
