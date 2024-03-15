from pydantic import BaseModel
from typing import Optional, Union, Dict, List, Any


class WSInput(BaseModel):
    command: str
    token: Optional[str] = None
    try_mode: Optional[bool] = None
    user_id: Optional[int] = None
    profile_id: Optional[int] = None
    story_id: Optional[int] = None
    conversation_id: Optional[str] = None
    session_ids: Optional[List[str]] = None
    message: Optional[Union[str, Dict[str, Any]]] = None
