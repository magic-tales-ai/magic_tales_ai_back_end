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

# Initialize config
with initialize(config_path="../config"):
    config = compose(config_name="config")

# local persistence
orchestrators = {}
users = {}


def update_dicts(websocket):
    global orchestrators, users
    if websocket in orchestrators:
        orchestrators.pop(websocket)
    users = {key: val for key, val in users.items() if val != websocket}

    logger.info(f"users: - {users}")
    logger.info(f"orchestrators: - {orchestrators}")


@ai_core_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info(f"users: {users}")
    logger.info(f"orchestrators: {orchestrators}")
    # TODO: remove this
    # Generate UID for the websocket instance ???
    websocket_uid = websocket.query_params.get(
        "uid",
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}_{str(uuid4()).upper()}",
    )

    websocket.uid = websocket_uid

    # Utility class for sending message to client
    # message_sender = MessageSender(websocket)

    # Asynchronously get a session for the duration of this websocket connection
    # TODO: check this...
    async for session in get_session():

        try:
            # Accept the websocket connection
            await websocket.accept()
            await websocket.send_json({"uid": websocket.uid})

            orchestrators[websocket] = MagicTalesCoreOrchestrator(
                config=config,
                message_sender=MessageSender(websocket),
                session=session,
                websocket=websocket,
            )

            logger.info(f"+ orchestrators: {orchestrators}")

            while True:
                try:
                    data = await websocket.receive_json()
                    request = WSInput(**data)

                    # Validate token
                    if not request.token:
                        await websocket.send_json(
                            {"uid": websocket.uid, "error": "Token is required."}
                        )
                        continue

                    try:
                        token_data = await check_token(request.token)
                    except:
                        # if not token_data:
                        await websocket.send_json(
                            {"uid": websocket.uid, "error": "Invalid token."}
                        )
                        continue
                    user_id = token_data["user_id"]
                    if user_id in users and not users[user_id] == websocket:
                        await websocket.send_json(
                            {"uid": websocket.uid, "error": "User connected already."}
                        )
                        continue

                    users[user_id] = websocket

                    logger.info(f"+ users: {users}")

                    # Validate command is not empty
                    if (
                        not request.command
                        in orchestrators[websocket].frontend_command_handlers
                    ):
                        await websocket.send_json(
                            {
                                "uid": websocket.uid,
                                "error": f"Command '{request.command}' invalid. Commands valids are: {' | '.join(i for i in orchestrators[websocket].frontend_command_handlers)}",
                            }
                        )
                        continue

                    # Process the command with the orchestrator
                    try:
                        await orchestrators[websocket].process_frontend_request(
                            request, token_data
                        )
                    except Exception as e:
                        await websocket.send_json(
                            {"uid": websocket.uid, "error": f"{traceback.format_exc()}"}
                        )

                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected by the server")
                    # TODO: remove refresh token
                    orchestrators[websocket]._cancel_token_refresh_task()
                    update_dicts(websocket)
                    # message_sender = None
                    break

        except Exception as ex:
            # await websocket.send_json({"error": str(ex)})
            # await websocket.close()
            logger.info(
                f"WebSocket connection error", exc_info=True
            )  #: {ex}/n/n{traceback.format_exc()}")
            update_dicts(websocket)
            # break  # Exit the loop to end the session context manager
        finally:
            if websocket.client_state.name != "DISCONNECTED":
                # remove data from user_data ... ???
                await websocket.close(code=1000)  # Normal closure
                update_dicts(websocket)
