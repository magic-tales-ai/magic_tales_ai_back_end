from fastapi import FastAPI
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env variables before db.py

import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from controllers.ai_core_controller import ai_core_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    
)

# Include routers
app.include_router(ai_core_router)

static_folder = os.getenv("STATIC_FOLDER")
app.mount(static_folder, StaticFiles(directory=static_folder), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("SERVER_HOST"),
        port=int(os.getenv("SERVER_PORT")),
        reload=True,
    )
