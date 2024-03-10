from pydantic import BaseModel
from typing import Optional, List


class WSOutput(BaseModel):
    uid: Optional[str] = None
    command: str
    token: Optional[str] = None
    data: Optional[dict] = None
    story_state: Optional[str] = None
    process_id: Optional[str] = None
    status: Optional[str] = None
    progress_percent: Optional[int] = None
    message: Optional[str] = None
    ack: Optional[bool] = None
    working: Optional[bool] = None
    images: Optional[List[Optional[str]]] = None
    files: Optional[List[Optional[str]]] = None

    def dict(self, **kwargs):
        # Delete None Properties
        data = super().dict(**kwargs)
        return {k: v for k, v in data.items() if v is not None and v != ""}
