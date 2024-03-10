from langchain.schema import HumanMessage, SystemMessage

from typing import Any, Dict

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


def initialize_image_prompt_response_dict(**kwargs) -> Dict[str, Any]:
    """
    Initialize the chapter information dictionary with default values or provided key-value pairs.

    :param kwargs: Optional key-value pairs to override the default values.
    :return: A dictionary containing the chapter information.
    """
    # Default chapter information dictionary
    image_prompt_response_dict = {
        "image_prompt_generator_success": True,
        "image_prompt_generator_prompt_messages": [
            SystemMessage(content=""),
            HumanMessage(content=""),
        ],
        "image_prompt_response_content_dict": {},
    }

    # Update the chapter information dictionary with the provided key-value pairs
    image_prompt_response_dict.update(kwargs)

    return image_prompt_response_dict
