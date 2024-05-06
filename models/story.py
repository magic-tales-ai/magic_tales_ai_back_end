import os
from typing import Any, Dict, List
import datetime
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

class StorySchema(Schema):
    id = fields.Int()
    profile_id = fields.Int()
    ws_session_uid = fields.Str()
    title = fields.Str()
    features = fields.Str()
    synopsis = fields.Str()
    last_successful_step = fields.Int()
    profile = fields.Nested(ProfileSchema, many=False)


@dataclass
class InMemStoryData:
    chapters: List[Dict[str, str]] = field(default_factory=list)
    num_chapters: int = field(default=0)
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
                logger.error(
                    f"Could not load InMemStoryData state from {filepath}. Error: {e}",
                    exc_info=True,
                )

        logger.info("Returning a new instance of InMemStoryData.")
        return InMemStoryData()  # Return a new instance if the load fails


def handle_custom_objects(obj):
    if isinstance(obj, StoryState):
        return obj.name
    elif isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")
