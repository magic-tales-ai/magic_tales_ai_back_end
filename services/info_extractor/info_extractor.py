import os
import json
import asyncio
import logging
from typing import Callable, Dict, List, Optional
from openai import AsyncOpenAI
from services.utils.log_utils import get_logger
from services.chat_assistant.prompt_utils import (
    async_load_prompt_template_from_file,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class InfoExtractor:
    def __init__(self, config):
        """
        Initialize the Magic Tales Information Extractor.

        Args:
            config (DictConfig): Configuration parameters.
        """
        logger.info("Initializing Information Extractor.")
        self._validate_openai_api_key()
        self.config = config
        self.client = AsyncOpenAI()
        self.info_extractor_assistant = None
        self.info_extractor_thread = None
        self.assistant_initialized = False

    def _validate_openai_api_key(self):
        """Validates the presence of the OpenAI API key."""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    async def _init_information_extractor_assistant(self):
        """
        Initializes the information extractor OpenAI assistant.
        """
        instructions = await async_load_prompt_template_from_file(
            self.config.instructions_path
        )
        self.info_extractor_assistant = await self.client.beta.assistants.create(
            name="Information Extractor",
            instructions=instructions,
            model=self.config.model,
        )
        self.info_extractor_thread = await self.client.beta.threads.create()
        self.assistant_initialized = True
        logger.info("Information Extractor OpenAI Assistant initialized.")

    async def extract_info(self, message: str) -> str:
        """
        Process the incoming message and generate an AI response.

        Args:
            message (str): The message to be processed.

        Returns:
            str: The response message from the AI assistant.
        """
        try:
            if not self.assistant_initialized:
                await self._init_information_extractor_assistant()

            await self.client.beta.threads.messages.create(
                thread_id=self.info_extractor_thread.id,
                role="user",
                content=message,
            )

            run = await self.client.beta.threads.runs.create(
                thread_id=self.info_extractor_thread.id,
                assistant_id=self.info_extractor_assistant.id,
            )

            while run.status in ["queued", "in_progress"]:
                await asyncio.sleep(0.5)
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=self.info_extractor_thread.id,
                    run_id=run.id,
                )

            messages = await self.client.beta.threads.messages.list(
                thread_id=self.info_extractor_thread.id, order="desc"
            )

            ai_message = self._find_latest_ai_response(messages.data)
            return ai_message.content[0].text.value if ai_message else ""
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return ""

    def _find_latest_ai_response(self, messages: List) -> Optional[dict]:
        """
        Find the latest AI response from a list of messages.

        Args:
            messages (List): A list of message dicts in reverse-chronological order.

        Returns:
            Optional[dict]: The latest message dict from the AI or None if not found.
        """
        for message in messages:
            if message.role == "assistant":
                return message
        return None
