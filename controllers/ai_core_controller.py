import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import traceback

from db import get_session  # get_session is NOW an async function
from uuid import uuid4
from datetime import datetime
from hydra import initialize, compose

from services.orchestrator.orchestrator import MagicTalesCoreOrchestrator
from services.message_sender import MessageSender
from services.session_service import check_token

from magic_tales_models.models.ws_input import WSInput
from services.utils.log_utils import get_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
# Get a logger instance for this module
logger = get_logger(__name__)

# TODO: change /bot route to /ai
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

                    # Validate token
                    if not request.token:
                        await websocket.send_json({"error": "Token is required."})
                        continue
                    try:
                        token_data = await check_token(request.token)
                    except:
                    #if not token_data:
                        await websocket.send_json({"error": "Invalid token."})
                        continue

                    # Validate command is not empty
                    if not request.command:
                        await websocket.send_json({"error": "Command can't be null"})
                        continue

                    # Process the command with the orchestrator
                    await orchestrator.process_frontend_request(request, token_data)

                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected by the server")
                    orchestrator._cancel_token_refresh_task()
                    orchestrator = None
                    message_sender = None
                    break

        except Exception as ex:
            # await websocket.send_json({"error": str(ex)})
            # await websocket.close()
            logger.info(f"WebSocket connection error", exc_info=True) #: {ex}/n/n{traceback.format_exc()}")
            # break  # Exit the loop to end the session context manager
        finally:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close(code=1000)  # Normal closure
