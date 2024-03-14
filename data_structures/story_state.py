from enum import Enum, auto


class EnhancedEnum(Enum):
    @classmethod
    def next(cls, current_state):
        """Returns the next state in the enum, or None if at the end."""
        members = list(cls)
        index = members.index(current_state)
        return members[index + 1] if index + 1 < len(members) else None

    def __str__(self):
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
