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
    "chapter_number": "The Chapter number we are working on (starting from 1).",
    "total_num_chapters": "The total number of chapters in the story.",
    "current_chapter_previous_version": "Current Chapter previous version (if not the first time attemting to generate this chapter).",
    "story_blueprint": "Information about the story, including the personality profile of the individual we are trying to target this story for, the story features (theme, genre, audience, etc), the story synopsis that we ageed with the buyer of the story, and the story title.",
    "previous_chapter": "Previous chapter within the story for context and alignment with the global plot.",
    "critique": "This is a list of all the things that the critique asked to be improved and why(rationale).",
}


def _get_input_format_description() -> str:
    """
    Constructs a description of the input format.

    Returns:
        str: A formatted string that describes the expected input format.
    """
    input_format_template = load_prompt("generator/input_format")
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
    rationale_schema = ResponseSchema(
        name="rationale",
        description="Explain how this chapter is aligned with the global plot and how it is aligned with the previous chapter.",
    )

    plan_schema = ResponseSchema(
        name="plan",
        description="Step by step plan, describing what this chapter will consist of in a succint way. Format: 1) ... 2) ... 3) ...",
    )

    synopsis_schema = ResponseSchema(
        name="synopsis",
        description="This is the chapter synopsis, which is a short description of what happens in the chapter.",
    )

    content_schema = ResponseSchema(
        name="content",
        description="""This is the chapter full content following your rationale, plan and synopsis.        
        """,
    )

    response_schemas = [
        # rationale_schema,
        plan_schema,
        # synopsis_schema,
        content_schema,
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
    system_message_template = load_prompt("generator/system_message")

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
    story_blueprint: Dict[str, str],
    previous_chapter_content: str,
    input_info: Dict[str, str],
) -> HumanMessage:
    """
    Constructs a human-readable message based on the given input information that will send to the AI.

    Args:
        input_info (Dict[str, str]): A dictionary containing input information.

    Returns:
        HumanMessage: The constructed human-readable message.
    """
    # Load the instructions template
    system_message_template = load_prompt("generator/input_format")
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(
        system_message_template
    )

    # Now, create the human message using the general info and the episodic_results
    human_message = human_message_prompt_template.format(
        chapter_number=input_info.get("chapter_number", 1),
        total_num_chapters=input_info.get("total_num_chapters", 1),
        current_chapter_previous_version=input_info.get(
            "chapter_generator_response_dict", {}
        ).get("content", ""),
        story_blueprint=story_blueprint,
        previous_chapter=previous_chapter_content,
        critique=input_info.get("chapter_critic_response_dict", {}),
    )

    return human_message
