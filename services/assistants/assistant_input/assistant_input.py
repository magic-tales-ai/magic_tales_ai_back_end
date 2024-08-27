from enum import Enum, auto


class Source(Enum):
    USER = auto()
    SYSTEM = auto()


class AssistantInput:
    def __init__(self, message: str, source: Source):
        self.message = message
        self.source = source

    async def to_json(self):
        return {"message": self.message, "source": self.source.name}
