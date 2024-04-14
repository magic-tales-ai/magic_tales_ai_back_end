from fastapi import WebSocket
from starlette.websockets import WebSocketState
import logging

from services.utils.log_utils import get_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)



class MessageSender:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send_message_to_frontend(self, message):
        message.uid = self.websocket.uid
        # await self.websocket.send_json(message.dict())
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.send_json(message.dict())
        else:
            logger.error("WebSocket connection is closed. Cannot send message.")
