from enum import Enum, auto
from typing import Optional, TypeVar

# Create a type variable that can be 'EnhancedEnum' or any subclass thereof.
E = TypeVar('E', bound='EnhancedEnum')

class EnhancedEnum(Enum):
    @classmethod
    def next(cls, current_state: E) -> Optional[E]:
        """Returns the next state in the enum, or None if at the end.
        
        Args:
            current_state (E): The current enum member.

        Returns:
            Optional[E]: The next enum member, or None if current is the last.
        """
        members = list(cls)
        index = members.index(current_state)
        return members[index + 1] if index + 1 < len(members) else None

    def __str__(self) -> str:
        """String representation for logging and readability."""
        return self.name

class StoryState(EnhancedEnum):
    USER_FACING_CHAT = auto()
    STORY_TITLE_GENERATION = auto()
    STORY_GENERATION = auto()
    IMAGE_PROMPT_GENERATION = auto()
    IMAGE_GENERATION = auto()
    DOCUMENT_GENERATION = auto()
    FINAL_DOCUMENT_GENERATED = auto()
