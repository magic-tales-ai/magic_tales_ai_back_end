import os
import markdown
import json
import logging
import traceback
from typing import Dict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.exc import SQLAlchemyError, NoResultFound
from models.story import Story, InMemStoryData
from models.profile import Profile
from models.story_state import StoryState
from services.utils.log_utils import get_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
# Get a logger instance for this module
logger = get_logger(__name__)


class StoryManager:
    def __init__(self, session: AsyncSession):
        self.session = session
        self.profile: Profile = None
        self.story: Story = None
        self.in_mem_story_data = InMemStoryData()

    async def load_story_manager(self, story_id: int):
        """Load story along with its chapters and images from the database."""

        try:
            logger.info("Loading all data into the story manager")
            self.story = await self.session.get(Story, story_id)
            if not self.story:
                raise ValueError(f"No story found with ID {story_id}")

            self.profile = await self.session.get(Profile, self.story.profile_id)
            if not self.profile:
                raise ValueError(f"No profile found with ID {self.story.profile_id}")

            self.in_mem_story_data = InMemStoryData.load_state(self.story.story_folder)
            logger.info("Story and profile loaded successfully")
        except Exception as e:
            logger.error(
                f"Failed to load story manager data: {e}\n\n{traceback.format_exc()}"
            )
            raise

    async def get_last_step(self):
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
        await self.session.commit()  # Ensure any pending transactions are committed.

    async def update_story(self, updates: dict):
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

        # Log the intended updates for audit and debugging purposes
        logger.info(
            f"Attempting to update story ID {self.story.id} with changes: {updates}"
        )

        try:
            for key, value in updates.items():
                if hasattr(self.story, key):
                    setattr(self.story, key, value)
                else:
                    logger.error(
                        f"Attempted to update non-existent attribute '{key}' on Story."
                    )
                    raise AttributeError(f"Story has no attribute '{key}'")

            await self.session.commit()
            await self.session.refresh(self.story)
            logger.info(f"Story ID {self.story.id} successfully updated.")
        except SQLAlchemyError as e:
            logger.error(
                f"Failed to update story ID {self.story.id}: {e}\n\n{traceback.format_exc()}"
            )
            await self.session.rollback()
            raise

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
            self.story.last_successful_step = step.value
            await self.session.commit()
            logger.info(f"Updated story ID {self.story.id} to step: {step.name}")
            
            # Refresh immediately after committing to sync state
            await self.session.refresh(self.story)
            
            # Save the in-memory story data
            try:
                self.in_mem_story_data.save_state(self.story.story_folder)
                logger.info("In-memory story data saved successfully.")
            except Exception as e:
                logger.error(f"Failed to save the in-memory story data: {e}\n\n{traceback.format_exc()}")
                raise Exception("Failed to save in-memory story data.") from e

        except SQLAlchemyError as e:
            logger.error(f"Failed to update story step: {e}\n\n{traceback.format_exc()}")
            await self.session.rollback()
            raise Exception("Database update failed, rolled back changes.") from e

    async def create_story(
        self,
        profile_id: int,
        session_id: str,
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
            session_id (str): Session identifier for the creation context.
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
            new_story = Story(
                profile_id=profile_id,
                session_id=session_id,
                title=title,
                features=features,
                synopsis=synopsis,
                story_folder=story_folder,
                images_subfolder=images_subfolder,
                last_successful_step=StoryState.USER_FACING_CHAT.value,
            )

            self.session.add(new_story)
            await self.session.commit()  # Commit the transaction to ensure the story is persisted
            await self.session.refresh(new_story)
            self.story = new_story  # Update the instance variable to the new story
            logger.info(f"Created new story with ID {self.story.id}")
        except SQLAlchemyError as e:
            await self.session.rollback()  # Rollback in case of an error
            logger.error(f"Failed to create story: {e}\n\n{traceback.format_exc()}")
            raise

    async def refresh(self):
        if self.story:
            await self.session.refresh(self.story)
        
        if self.profile:
            await self.session.refresh(self.profile)

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
            raise RuntimeError("Story or profile not loaded")

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
