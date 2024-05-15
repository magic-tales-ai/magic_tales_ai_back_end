import os
import markdown
import json
import logging
import traceback
from typing import Dict, Optional
import datetime

<<<<<<< HEAD
from magic_tales_models.models.story import Story, InMemStoryData
from magic_tales_models.models.profile import Profile
from magic_tales_models.models.story_state import StoryState
=======
from magic_tales_models.models.story import Story
from models.story import InMemStoryData
from magic_tales_models.models.profile import Profile
from models.story_state import StoryState
>>>>>>> dco/db-models-changes
from .database_manager import DatabaseManager
from services.utils.log_utils import get_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
# Get a logger instance for this module
logger = get_logger(__name__)


class StoryManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.profile: Optional[Profile] = None
        self.story: Optional[Story] = None
        self.in_mem_story_data = InMemStoryData()

    async def load_story(self, story_id: int):
        """Load story along with its chapters and images from the database."""

        try:
            logger.info("Loading all data into the story manager")
            self.story = await self.db_manager.fetch_story_by_id(story_id)
            if not self.story:
                raise ValueError(f"No story found with ID {story_id}")

            self.profile = await self.db_manager.fetch_profile_by_id(
                self.story.profile_id
            )
            if not self.profile:
                raise ValueError(f"No profile found with ID {self.story.profile_id}")

            self.in_mem_story_data = InMemStoryData.load_state(self.story.story_folder)
            logger.info("Story and profile loaded successfully")
        except Exception as e:
            logger.error(
                f"Failed to load story manager data: {e}\n\n{traceback.format_exc()}"
            )
            raise

    async def get_last_step(self) -> StoryState:
        """Function to get the last successful step as a StoryState enum."""
        try:
            # Check if the last_successful_step is a valid StoryState
            return StoryState(self.story.last_successful_step)
        except ValueError:
            # If last_successful_step is not valid, default to USER_FACING_CHAT
            return StoryState.USER_FACING_CHAT

    async def reset(self):
        """Reset the manager to clear any loaded data and prepare for a new story or cleanup."""
        self.story: Story = None
        self.profile: Profile = None
        self.in_mem_story_data = InMemStoryData()

    async def update_story(self, updates: Dict[str, any]):
        """
        Update the story's data based on provided updates dictionary.

        Args:
            updates (dict): A dictionary containing keys and values to update on the story.

        Raises:
            RuntimeError: Raised if no story is loaded before attempting to update.
            AttributeError: Raised if an invalid attribute is passed in updates dictionary.
        """
        if not self.story:
            raise RuntimeError("No story loaded to update.")
        try:
            logger.info(
                f"Attempting to update story ID {self.story.id} with changes: {updates}"
            )
            updates["last_updated"] = datetime.datetime.now(datetime.UTC)
            await self.db_manager.update_story_record_by_id(self.story.id, updates)
            logger.info(f"Story ID {self.story.id} successfully updated.")
        except Exception as e:
            logger.error(f"Failed to update the story: {e}\n\n{traceback.format_exc()}")
            raise Exception("Failed to update the story.") from e

    async def update_story_step(self, step: StoryState):
        """
        Update the story's last successful step in the database and save the in-memory story data.

        Args:
            step (StoryState): The step to update the story's last successful step to.

        Raises:
            RuntimeError: If no story is loaded.
            Exception: If there are issues updating the database or saving the state.
        """
        if not self.story:
            raise RuntimeError("No story loaded to update step.")

        try:
            updates = {
                "last_successful_step": step.value,
                "last_updated": datetime.datetime.now(datetime.UTC),
            }
            await self.db_manager.update_story_record_by_id(self.story.id, updates)
            logger.info(f"Updated story ID {self.story.id} to step: {step.name}")

            # Save the in-memory story data
            try:
                self.in_mem_story_data.save_state(self.story.story_folder)
                logger.info("In-memory story data saved successfully.")
            except Exception as e:
                logger.error(
                    f"Failed to save the in-memory story data: {e}\n\n{traceback.format_exc()}"
                )
                raise Exception("Failed to save in-memory story data.") from e

        except Exception as e:
            logger.error(
                f"Failed to update the story step: {e}\n\n{traceback.format_exc()}"
            )
            raise Exception("Failed to update the story step.") from e

    async def create_story(
        self,
        profile_id: int,
        ws_session_uid: str,
        title: str,
        features: str,
        synopsis: str,
        story_folder: str,
        images_subfolder: str,
    ) -> None:
        """
        Create a new story and save it to the database with robust error handling.

        Args:
            profile_id (int): ID of the profile to which the story belongs.
            ws_session_uid (str): WebSocket Session identifier for the creation context.
            title (str): Title of the story.
            features (str): Descriptive features of the story.
            synopsis (str): Synopsis of the story.
            story_folder (str): Folder where we will host all files for the story
            images_subfolder (str): Images subfolder (usually "images") where we will host all the images for the story

        Returns:
            Story: The newly created story instance.

        Raises:
            Exception: If the database transaction fails.
        """
        try:
            new_story = await self.db_manager.create_story(
                profile_id,
                ws_session_uid,
                title,
                features,
                synopsis,
                story_folder,
                images_subfolder,
            )
            await self.load_story(new_story.id)
            self.story = new_story  # Update the instance variable to the new story
            logger.info(f"Created new story with ID {self.story.id}")

        except Exception as e:
            logger.error(
                f"Failed to create a new story: {e}\n\n{traceback.format_exc()}"
            )
            raise Exception("Failed to create a new story") from e

    async def refresh(self, raise_error: bool = False):
        """
        Attempts to refresh the story and profile objects from the database.
        Includes comprehensive error handling and logging to ensure integrity
        and availability of these objects post-refresh.
        """
        try:
            if self.story:
                await self.db_manager.refresh_story(self.story)
                if self.story is None:  # Validating post-refresh
                    logging.warning("Story remains None after attempting to refresh.")
                else:
                    logging.info(
                        f"Story with ID {self.story.id} refreshed successfully."
                    )
            else:
                logging.warning("No story to refresh.")

            if self.profile:
                await self.db_manager.refresh_profile(self.profile)
                if self.profile is None:  # Validating post-refresh
                    logging.warning("Profile remains None after attempting to refresh.")
                else:
                    logging.info(
                        f"Profile with ID {self.profile.id} refreshed successfully."
                    )
            else:
                logging.warning("No profile to refresh.")
        except Exception as e:
            logging.error(f"Failed to refresh story or profile: {e}")
            raise RuntimeError(
                f"Refresh failed due to an error with the database manager: {e}"
            ) from e

        if raise_error and (not self.story or not self.profile):
            raise RuntimeError(
                "Critical refresh error: Story or Profile is still not loaded correctly after refresh."
            )

    async def get_story_blueprint(self) -> Dict[str, str]:
        """
        Retrieves the initial data necessary for generating a new story. This method extracts
        essential elements such as title, features, and synopsis from the loaded story and
        the recipient's details from the associated profile. This 'blueprint' serves as the foundation
        for the story generation process.

        Returns:
            Dict[str, str]: A dictionary containing key components of the story blueprint, including
                            the target recipient's details, story title, features, and synopsis.

        Raises:
            RuntimeError: If the story or profile has not been loaded prior to calling this method.
        """
        if not self.story or not self.profile:
            try:
                await self.refresh()
            except Exception as e:
                logger.error(
                    f"Failed to get the story blueprint: {e}\n\n{traceback.format_exc()}"
                )
                raise Exception("Failed to get the story blueprint") from e

        # Additional checks after refresh attempt to ensure that both self.story and self.profile are not None
        if not self.story:
            raise RuntimeError("Story is not loaded and cannot generate blueprint.")
        if not self.profile:
            raise RuntimeError("Profile is not loaded and cannot generate blueprint.")

        return {
            "target_recipient_of_the_story": self.profile.details,
            "story_title": self.story.title,
            "story_features": self.story.features,
            "story_synopsis": self.story.synopsis,
        }

    def export_to_human_readable_format(self) -> None:
        """
        Export the content of the current instance into human-readable files.

        Args:
            None

        Returns:
            None
        """
        # Create the main story folder
        story_folder_path = os.path.join(self.story.story_folder, "Story_Data")
        os.makedirs(story_folder_path, exist_ok=True)

        # Personality Profile
        with open(os.path.join(story_folder_path, "Personality_Profile.md"), "w") as f:
            f.write(markdown.markdown(self.profile.details))

        # Story Features
        with open(os.path.join(story_folder_path, "Story_Features.md"), "w") as f:
            f.write(markdown.markdown(self.story.features))

        # Synopsis
        with open(os.path.join(story_folder_path, "Synopsis.md"), "w") as f:
            f.write(markdown.markdown(self.story.synopsis))

        # Title
        if self.story.title:
            with open(os.path.join(story_folder_path, "Title.txt"), "w") as f:
                f.write(self.story.title)

        # Image Prompt Generator Messages
        md = "## Image Prompt Generator Messages\n"
        for idx, message in enumerate(
            self.in_mem_story_data.image_prompt_generator_prompt_messages
        ):
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
        for idx, chapter in enumerate(self.in_mem_story_data.post_processed_chapters):
            with open(
                os.path.join(
                    post_processed_folder, f"Post_Processed_Chapter_{idx + 1}.json"
                ),
                "w",
            ) as f:
                json.dump(chapter, f, indent=4)

        # Image Prompts
        image_prompts_folder = os.path.join(
            story_folder_path, self.story.images_subfolder
        )
        os.makedirs(image_prompts_folder, exist_ok=True)
        for idx, image_prompt in enumerate(self.in_mem_story_data.image_prompts):
            with open(
                os.path.join(image_prompts_folder, f"Image_Prompt_{idx + 1}.json"), "w"
            ) as f:
                json.dump(image_prompt, f, indent=4)

        # Image Filenames
        with open(os.path.join(story_folder_path, "Image_Filenames.txt"), "w") as f:
            for filename in self.in_mem_story_data.image_filenames:
                f.write(f"{filename}\n")
