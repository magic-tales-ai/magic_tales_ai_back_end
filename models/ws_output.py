from pydantic import BaseModel
from typing import Optional, Union, Dict, List, Any
from pydantic.json import pydantic_encoder
import json
from enum import Enum


def custom_json_encoder(obj):
    if isinstance(obj, Enum):
        return obj.value  # or str(obj) depending on your preference
    # Fallback to Pydantic's default encoder for other types
    return pydantic_encoder(obj)


class WSOutput(BaseModel):
    uid: Optional[str] = None
    command: str
    token: Optional[str] = None
    data: Optional[dict] = None
    story_state: Optional[str] = None
    process_id: Optional[str] = None
    status: Optional[str] = None
    progress_percent: Optional[int] = None
    message: Optional[Union[str, Dict[str, Any]]] = None
    ack: Optional[bool] = None
    working: Optional[bool] = None
    images: Optional[List[Optional[str]]] = None
    files: Optional[List[Optional[str]]] = None

    # def dict(self, **kwargs):
    #     # Delete None Properties
    #     data = super().dict(**kwargs)
    #     return {k: v for k, v in data.items() if v is not None and v != ""}

    def dict(self, **kwargs):
        return json.loads(
            json.dumps(super().dict(**kwargs), default=custom_json_encoder)
        )
