
import traceback
from typing import List, Optional
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.future import select
from sqlalchemy.sql import delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import desc

from models.user import User
from models.profile import Profile
from models.story import Story
from models.message import Message


class DatabaseManager:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def fetch_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Asynchronously retrieves a User by their ID.
        
        Args:
            user_id (int): The ID of the user to retrieve.
        
        Returns:
            Optional[User]: The User object if found, else None.

        Raises:
            Exception: If there is a database error during the operation.
        """
        try:
            return await self.session.get(User, user_id)
        except SQLAlchemyError as e:
            raise Exception(f"Failed to retrieve user by ID: {user_id}. Error: {e}")

    async def fetch_profile_by_id(self, profile_id: int) -> Optional[Profile]:
        """
        Asynchronously retrieves a Profile by its ID.
        
        Args:
            profile_id (int): The ID of the profile to retrieve.
        
        Returns:
            Optional[Profile]: The Profile object if found, else None.

        Raises:
            Exception: If there is a database error during the operation.
        """
        try:
            return await self.session.get(Profile, profile_id)
        except SQLAlchemyError as e:
            raise Exception(f"Failed to retrieve profile by ID: {profile_id}. Error: {e}")

    async def fetch_story_by_id(self, story_id: int) -> Optional[Story]:
        """
        Asynchronously retrieves a Story by its ID.
        
        Args:
            story_id (int): The ID of the story to retrieve.
        
        Returns:
            Optional[Story]: The Story object if found, else None.

        Raises:
            Exception: If there is a database error during the operation.
        """
        try:
            return await self.session.get(Story, story_id)
        except SQLAlchemyError as e:
            raise Exception(f"Failed to retrieve story by ID: {story_id}. Error: {e}")

    async def fetch_stories_for_user(self, user_id: int) -> List[Story]:
        """
        Asynchronously retrieves all Stories associated with a User's ID, through their Profiles.
        
        Args:
            user_id (int): The ID of the user whose stories are to be retrieved.
        
        Returns:
            List[Story]: A list of Story objects associated with the user.

        Raises:
            Exception: If there is a database error during the operation.
        """
        try:
            profiles = await self.fetch_profiles_for_user(user_id)
            profiles_ids = [profile.id for profile in profiles]

            if profiles_ids:
                stories_result = await self.session.execute(
                    select(Story).filter(Story.profile_id.in_(profiles_ids))
                )
                return stories_result.scalars().all()
            return []
        except SQLAlchemyError as e:
            raise Exception(f"Failed to retrieve stories for user ID: {user_id}. Error: {e}")
        
    async def fetch_profiles_for_user(self, user_id: int) -> List[Profile]:
        """
        Asynchronously retrieves all Profiles associated with a User's ID.
        
        Args:
            user_id (int): The ID of the user whose profiles are to be retrieved.
        
        Returns:
            List[Profile]: A list of Profile objects associated with the user, or an empty list if no profiles are found.

        Raises:
            Exception: If there is a database error during the operation.
        """
        try:
            result = await self.session.execute(
                select(Profile).where(Profile.user_id == user_id)
            )
            profiles = result.scalars().all()
            if not profiles:
                return []
            return profiles
        except SQLAlchemyError as e:
            raise Exception(f"Failed to retrieve profiles for user ID: {user_id}. Error: {e}")
        
    async def fetch_stories_for_profile(self, profile_id: int) -> List[Story]:
        """
        Asynchronously retrieves all Stories by a specific Profile ID.
        
        Args:
            profile_id (int): The ID of the profile whose stories are to be retrieved.
        
        Returns:
            List[Story]: A list of Story objects associated with the profile.

        Raises:
            Exception: If there is a database error during the operation.
        """
        try:
            result = await self.session.execute(
                select(Story).where(Story.profile_id == profile_id)
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to retrieve stories by profile ID: {profile_id}. Error: {e}")

    async def fetch_messages_for_session(self, session_id: str) -> List[Message]:
        """
        Asynchronously retrieves all Messages by a Session ID, specifically filtering for 'chat' type messages.
        
        Args:
            session_id (str): The session ID to filter messages by.
        
        Returns:
            List[Message]: A list of Message objects filtered by session ID and type.

        Raises:
            Exception: If there is a database error during the operation.
        """
        try:
            result = await self.session.execute(
                select(Message)
                .where(Message.session_id == session_id, Message.type == "chat")
                .order_by(desc(Message.session_id), Message.id)
            )
            return result.scalars().all()
        except SQLAlchemyError as e:
            raise Exception(f"Failed to retrieve messages by session ID: {session_id}. Error: {e}")

    async def create_profile_from_chat_details(self, user_id: int, profile_details: str) -> Profile:
        """
        Asynchronously creates a profile from chat details provided.

        Args:
            user_id (int): User ID to which the profile will be linked.
            profile_details (str): Details extracted from the chat to be used as profile description.

        Returns:
            Profile: The newly created profile object.

        Raises:
            Exception: If profile creation fails.
        """
        try:
            new_profile = Profile(user_id=user_id, details=profile_details)
            self.session.add(new_profile)
            await self.session.commit()
            await self.session.refresh(new_profile)
            return new_profile
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise Exception(f"Failed to create profile: {traceback.format_exc()}") from e

    async def create_user_record(self, new_user: User) -> User:
        """
        Asynchronously creates a new user in the database.

        Args:
            new_user (User): The user data to be added.

        Returns:
            User: The newly created user with ID updated.

        Raises:
            Exception: If user creation fails.
        """
        try:
            self.session.add(new_user)
            await self.session.commit()
            await self.session.refresh(new_user)
            return new_user
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise Exception(f"Failed to create user: {traceback.format_exc()}") from e

    async def add_message_to_session_record(self, message: Message) -> Message:
        """
        Asynchronously adds a new message to a session in the database.

        Args:
            message (Message): The message data to be added.

        Returns:
            Message: The newly added message with ID updated.

        Raises:
            Exception: If adding the message fails.
        """
        try:
            self.session.add(message)
            await self.session.commit()
            await self.session.refresh(message)
            return message
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise Exception(f"Failed to add message to session: {traceback.format_exc()}") from e

    async def delete_messages_by_session(self, session_id: int):
        """
        Asynchronously deletes messages by session ID.

        Args:
            session_id (int): The session ID whose messages are to be deleted.

        Raises:
            Exception: If message deletion fails.
        """
        try:
            await self.session.execute(delete(Message).where(Message.session_id == session_id))
            await self.session.commit()
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise Exception(f"Failed to delete messages: {traceback.format_exc()}") from e

    async def update_user_record_by_id(self, user_id: int, data: dict) -> User:
        """
        Asynchronously updates a user by ID with provided data.

        Args:
            user_id (int): The user ID to update.
            data (dict): A dictionary of attributes to update.

        Returns:
            User: The updated user object.

        Raises:
            Exception: If user update fails.
        """
        try:
            user = await self.session.get(User, user_id)
            for key, value in data.items():
                setattr(user, key, value)
            await self.session.commit()
            await self.session.refresh(user)
            return user
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise Exception(f"Failed to update user: {traceback.format_exc()}") from e

    async def update_profile_record_by_id(self, profile_id: int, data: dict) -> Profile:
        """
        Asynchronously updates a profile by ID with provided data.

        Args:
            profile_id (int): The profile ID to update.
            data (dict): A dictionary of attributes to update.

        Returns:
            Profile: The updated profile object.

        Raises:
            Exception: If profile update fails.
        """
        try:
            profile = await self.session.get(Profile, profile_id)
            for key, value in data.items():
                setattr(profile, key, value)
            await self.session.commit()
            await self.session.refresh(profile)
            return profile
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise Exception(f"Failed to update profile: {traceback.format_exc()}") from e

    async def link_user_with_conversations_by_session_ids(self, user_id: int, session_ids: list):
        """
        Asynchronously links a user with conversations by updating message records.

        Args:
            user_id (int): The user ID to link with messages.
            session_ids (list): A list of session IDs whose messages will be linked to the user.

        Raises:
            Exception: If linking fails.
        """
        try:
            await self.session.execute(
                update(Message)
                .where(Message.session_id.in_(session_ids))
                .values(user_id=user_id)
            )
            await self.session.commit()
        except SQLAlchemyError as e:
            await self.session.rollback()
            raise Exception(f"Failed to link user with conversations: {traceback.format_exc()}") from e