from fastapi import FastAPI
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env variables before db.py

import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from controllers.ai_core_controller import ai_core_router

# import newrelic.agent
# newrelic.agent.initialize('newrelic.ini')
# application = newrelic.agent.register_application(timeout=10.0)

import sentry_sdk

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
DEFAULT_STATIC_FOLDER = "/tmp"

sentry_sdk.init(
    dsn="https://28c83116aebb31f9bdb4619a95ae5135@o4507544276303872.ingest.us.sentry.io/4507544617680896",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ai_core_router)

static_folder = os.getenv("STATIC_FOLDER", DEFAULT_STATIC_FOLDER)
app.mount(static_folder, StaticFiles(directory=static_folder), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("SERVER_HOST", DEFAULT_SERVER_HOST),
        port=int(os.getenv("SERVER_PORT", DEFAULT_SERVER_PORT)),
        proxy_headers=True,
        forwarded_allow_ips="*",
        reload=True,
    )
