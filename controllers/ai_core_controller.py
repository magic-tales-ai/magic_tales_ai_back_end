from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import traceback

from db import get_session  # get_session is NOW an async function
from uuid import uuid4
from datetime import datetime
from hydra import initialize, compose

from services.orchestrator.orchestrator import MagicTalesCoreOrchestrator
from services.message_sender import MessageSender
from services.session_service import check_token

from models.ws_input import WSInput

ai_core_router = APIRouter(prefix="/bot", tags=["Bot"])


@ai_core_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Initialize config
    with initialize(config_path="../config"):
        config = compose(config_name="config")

    # Generate UID for the websocket instance
    websocket_uid = websocket.query_params.get(
        "uid",
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}_{str(uuid4()).upper()}",
    )
    websocket.uid = websocket_uid

    # Utility class for sending message to client
    message_sender = MessageSender(websocket)

    # Asynchronously get a session for the duration of this websocket connection
    async for session in get_session():
        # Orchestrator: main class
        orchestrator = MagicTalesCoreOrchestrator(
            config=config,
            message_sender=message_sender,
            session=session,
            websocket=websocket,
        )

        try:
            # Accept the websocket connection
            await websocket.accept()
            await websocket.send_json({"uid": websocket.uid})

            while True:
                try:
                    data = await websocket.receive_json()
                    request = WSInput(**data)

                    # Validate token if not in try_mode
                    if not request.try_mode:
                        token_data = await check_token(request.token)
                        if not token_data:
                            raise Exception("Invalid token.")
                    else:
                        token_data = None

                    # Validate command is not empty
                    if not request.command:
                        await websocket.send_json({"error": "Command can't be null"})
                        continue

                    # Process the command with the orchestrator
                    await orchestrator.process_request(request, token_data)

                except WebSocketDisconnect:
                    print("WebSocket disconnected by the server")
                    break

        except Exception as ex:
            # await websocket.send_json({"error": str(ex)})
            # await websocket.close()
            print(f"WebSocket connection error: {ex}/n/n{traceback.format_exc()}")
            # break  # Exit the loop to end the session context manager
        finally:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close(code=1000)  # Normal closure
