from fastapi import WebSocket


class MessageSender:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def send_message_to_frontend(self, message):
        message.uid = self.websocket.uid
        await self.websocket.send_json(message.dict())
