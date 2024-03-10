from fastapi import APIRouter, WebSocket, Depends
from db import get_session
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
from hydra import initialize, compose

from services.orchestrator.magic_tales_orchestrator import MagicTalesCoreOrchestrator
from services.MessageSender import MessageSender
from services.SessionService import check_token

from data_structures.ws_input import WSInput

ai_core_router = APIRouter(prefix="/bot", tags=["Bot"])


@ai_core_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, session: Session = Depends(get_session)
):

    with initialize(config_path="../config"):
        config = compose(config_name="config")

    # each WS connection generates its own instance
    # each instance has its own components

    # Each websocket instance has its own instance UID
    # UID composes by date and a generated GUID
    if websocket.query_params and "uid" in websocket.query_params:
        websocket.uid = websocket.query_params["uid"]
    else:
        actual_date = (
            str(datetime.now()).replace(":", "-").replace(" ", "-").replace(".", "-")
        )
        websocket_uid = actual_date + "-" + str(uuid4()).upper()
        websocket.uid = websocket_uid
    # message sender: utility class for sending message to client
    message_sender = MessageSender(websocket)
    # orchestrator: main class
    orchestrator = MagicTalesCoreOrchestrator(
        config=config,
        message_sender=message_sender,
        session=session,
        websocket=websocket,
    )

    try:
        # Accept websocket
        await websocket.accept()
        await websocket.send_json({"uid": websocket.uid})

        while True:
            data = await websocket.receive_json()
            input_model = WSInput(**data)

            # each received message needs TOKEN parameter
            # token is used to validate SESSION (API SESSION)
            if input_model.try_mode is False or input_model.try_mode is None:
                token_data = await check_token(input_model.token)
                if not (token_data):
                    raise Exception("Invalid token.")
            # if try_mode is True
            else:
                token_data = None

            # each received message needs COMMAND parameter
            if input_model.command is None or input_model.command == "":
                await websocket.send_json({"error": "Command can't be null"})

            # ANY commands GOES to ORCHESTRATOR
            await orchestrator.process(input_model, token_data)

    except Exception as ex:
        await websocket.send_json({"error": f"{ex}"})
        await websocket.close()
        print(f"WebSocket connection error: {ex}")
