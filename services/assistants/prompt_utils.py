import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


async def async_load_prompt_template_from_file(file_path: Path) -> str:
    with open(file_path, "r") as file:
        prompt = file.read()
    return prompt


def load_prompt_template_from_file(file_path: Path) -> str:
    with open(file_path, "r") as file:
        prompt = file.read()
    return prompt
