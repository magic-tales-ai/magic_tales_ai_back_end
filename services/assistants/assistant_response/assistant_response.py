from typing import Optional


class AssistantResponse:
    def __init__(self, message_for_user: str, error: Optional[str] = None):
        self.message_for_user = message_for_user
        self.error = error

    async def has_error(self) -> bool:
        """
        Check if the response contains an error.

        Returns:
            bool: True if there is an error, otherwise False.
        """
        return bool(self.error)

    async def serialize(self):
        """
        Serialize the response to a suitable format that can be used by the client or logged.

        Returns:
            Dict: A dictionary representation of the message and error.
        """
        return {"message_for_user": self.message_for_user, "error": self.error}

    async def get_message_for_user(self) -> str:
        return self.message_for_user

    async def get_error(self) -> Optional[str]:
        return self.error
