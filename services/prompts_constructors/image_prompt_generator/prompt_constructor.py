import re
from typing import Dict, Tuple

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import (
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

from services.prompts_constructors.utils.prompt_utils import load_prompt

# Define input descriptions at the module level
INPUT_DESCRIPTIONS = {
    "chapter_number": "The Chapter number we are working on.",
    "chapter_content": "The Chapter content we are working on.",
}


def _get_input_format_description() -> str:
    """
    Constructs a description of the input format.

    Returns:
        str: A formatted string that describes the expected input format.
    """
    input_format_template = load_prompt("image_prompt_generator/input_format")
    prompt_template = PromptTemplate.from_template(input_format_template)
    return prompt_template.format(**INPUT_DESCRIPTIONS)


def _get_response_format_instructions_prompt_and_output_parser() -> (
    Tuple[str, StructuredOutputParser]
):
    """
    Constructs a description of the response format and prepares an output parser.

    Returns:
        Tuple[str, StructuredOutputParser]: A tuple containing a formatted string that describes the expected response format and a parser for structured output.
    """
    # Prepare the schemas for each response field
    image_prompts = ResponseSchema(
        name="image_prompts",
        description="list of rich image prompts for an AI image Generator Bot using the prompt generation hints provided.",
        type="List[str]",
    )

    annotated_chapter_schema = ResponseSchema(
        name="annotated_chapter",
        description="""This is the exact same chapter content but with the addition of annotations in the form of '[img: <chapter_number>.<image_prompt_index>]' 
        to indicate that an image should be inserted at that point in the story. Make sure you add an annotation for each image prompt you are creating and in the order they are created.
        For example, your first image prompt on the image_prompts List should be annotated as '[img: <chapter_number>.0]', the second as '[img: <chapter_number>.1]', and so on.
        """,
    )

    response_schemas = [
        image_prompts,
        annotated_chapter_schema,
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    return output_parser.get_format_instructions(), output_parser


def _get_system_message_prompt_template_and_parser() -> (
    Tuple[SystemMessagePromptTemplate, StructuredOutputParser]
):
    """
    Constructs a system message prompt template and prepares an output parser.

    Returns:
        Tuple[SystemMessagePromptTemplate, StructuredOutputParser]: A tuple containing a system message prompt template and a parser for structured output.
    """
    system_message_template = load_prompt("image_prompt_generator/system_message")

    # Prepare the main prompt template
    main_prompt_template = PromptTemplate.from_template(
        system_message_template,
        partial_variables={
            "input_format_description": _get_input_format_description(),
            "response_format_instructions": _get_response_format_instructions_prompt_and_output_parser()[
                0
            ],
        },
    )

    return (
        SystemMessagePromptTemplate(prompt=main_prompt_template),
        _get_response_format_instructions_prompt_and_output_parser()[1],
    )


def compose_system_message_and_create_output_parser() -> (
    Tuple[SystemMessage, StructuredOutputParser]
):
    """
    Constructs a system message and prepares an output parser based on the given storys.

    Args:
        storys str: A str containing storys.

    Returns:
        Tuple[SystemMessage, StructuredOutputParser]: A tuple containing a system message and a parser for structured output.
    """
    # Get the system message prompt template and the output parser
    (
        system_message_prompt_template,
        output_parser,
    ) = _get_system_message_prompt_template_and_parser()

    # Format the system message
    system_message = system_message_prompt_template.format()

    return system_message, output_parser


def compose_message_for_LLM(
    chapter_number: str,
    chapter_content: str,
) -> HumanMessage:
    """
    Constructs a human-readable message based on the given input information that will send to the AI.

    Args:
        input_info (Dict[str, str]): A dictionary containing input information.

    Returns:
        HumanMessage: The constructed human-readable message.
    """
    # Load the instructions template
    system_message_template = load_prompt("image_prompt_generator/input_format")
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(
        system_message_template
    )

    # Now, create the human message using the general info and the episodic_results
    human_message = human_message_prompt_template.format(
        chapter_number=chapter_number,
        chapter_content=chapter_content,
    )

    return human_message
