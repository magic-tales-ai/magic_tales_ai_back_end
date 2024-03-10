import os
from typing import Dict

from langchain_community.chat_models.openai import BaseChatModel
from langchain.output_parsers import (
    OutputFixingParser,
    RetryWithErrorOutputParser,
    StructuredOutputParser,
)

from services.utils.file_utils import load_text

PROMPTS_PATH = os.path.join(os.getcwd(), "magic_tales/prompts_constructors")


def load_prompt(prompt: str, package_path: str = PROMPTS_PATH) -> str:
    """
    Load a text file given the prompt name.

    Args:
        prompt (str): The name of the prompt to load.
        package_path (str, optional): The path to the directory containing the prompts. Defaults to PROMPTS_PATH.

    Returns:
        str: The contents of the prompt text file.
    """
    return load_text(os.path.join(package_path, f"{prompt}.txt"))


def get_retry_output_parser(
    output_parser: StructuredOutputParser, llm: BaseChatModel
) -> RetryWithErrorOutputParser:
    """
    Get a RetryWithErrorOutputParser instance.

    Args:
        output_parser (StructuredOutputParser): The structured output parser.
        llm (BaseChatModel): The language model.

    Returns:
        RetryWithErrorOutputParser: The RetryWithErrorOutputParser instance.
    """
    # Tries to fix parsing errors
    output_fixing_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm)
    # Requests new llm response if previous was broken
    retry_output_parser = RetryWithErrorOutputParser.from_llm(
        parser=output_fixing_parser, llm=llm
    )
    return retry_output_parser
