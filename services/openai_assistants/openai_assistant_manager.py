import asyncio
import openai
import os
from typing import List, Dict
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable
import logging

from services.utils.log_utils import get_logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class OpenAIAssistantManager:
    def __init__(self):
        self._validate_openai_api_key()
        self.client = openai.AsyncOpenAI()

    def _validate_openai_api_key(self):
        """Validates the presence of the OpenAI API key."""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")

    @asynccontextmanager
    async def _safe_api_call(self, api_call: Callable[..., AsyncGenerator[None, None]]):
        """A context manager to safely make API calls with error handling and logging."""
        try:
            yield await api_call()
        except Exception as e:
            logger.error(f"API call failed: {e}", exc_info=True)
            raise

    async def list_all_assistants(self) -> List[Dict]:
        """Lists all OpenAI Assistants."""
        async with self._safe_api_call(self.client.beta.assistants.list) as response:
            return response.data

    def format_assistants_list(self, assistants_response):
        print("OpenAI Assistants Overview:\n")
        for assistant in assistants_response:
            print(f"Name: {assistant.name}")
            print(f"ID: {assistant.id}")
            print(f"Model: {assistant.model}")
            if assistant.description:
                print(f"Description: {assistant.description}")
            if assistant.instructions:
                print("Instructions:")
                # Assuming instructions might contain newlines for better formatting
                print(assistant.instructions.replace("\\n", "\n"))
            if assistant.file_ids:
                print("Attached File IDs:")
                for file_id in assistant.file_ids:
                    print(f" - {file_id}")
            print("Tools:")
            for tool in assistant.tools:
                print(f" - Type: {tool.type}")
            print("\n-------------------\n")

    async def list_all_files_for_assistant(self, assistant_id: str) -> List[str]:
        """Lists all files for a given assistant."""
        files = self.client.beta.assistants.files.list(assistant_id=assistant_id)
        file_ids = [file["id"] for file in files["data"]]
        return file_ids

    async def delete_files_for_assistant(self, assistant_id: str, file_ids: List[str]):
        """Deletes all files for a given assistant."""
        for file_id in file_ids:
            self.client.beta.assistants.files.delete(
                assistant_id=assistant_id, file_id=file_id
            )

    async def delete_assistant(self, assistant_id: str):
        """Deletes a given assistant."""
        self.client.beta.assistants.delete(assistant_id=assistant_id)

    async def cleanup_all_assistants(self):
        """Deletes all assistants and their files."""
        assistant_ids = await self.list_all_assistants()
        for assistant_id in assistant_ids:
            file_ids = await self.list_all_files_for_assistant(
                assistant_id=assistant_id
            )
            await self.delete_files_for_assistant(assistant_id, file_ids)
            await self.delete_assistant(assistant_id)
        print("All assistants and their files have been deleted.")


if __name__ == "__main__":

    async def main():
        manager = OpenAIAssistantManager()
        # Example usage
        assistants = await manager.list_all_assistants()
        manager.format_assistants_list(assistants)
        await manager.cleanup_all_assistants()
        # Extend with more examples as needed.

    asyncio.run(main())
