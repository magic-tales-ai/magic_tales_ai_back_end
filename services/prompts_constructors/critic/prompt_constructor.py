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
    "current_chapter": "Current Chapter previous version (if not the first time attemting to generate this chapter).",
    "story_blueprint": "Information about the story, including the personality profile of the individual we are trying to target this story for, the story features (theme, genre, audience, etc), the story synopsis that we ageed with the buyer of the story, and the story title.",
    "previous_chapter": "Previous chapter within the story for context and alignment with the global plot.",
}


def _get_input_format_description() -> str:
    """
    Constructs a description of the input format.

    Returns:
        str: A formatted string that describes the expected input format.
    """
    input_format_template = load_prompt("critic/input_format")
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
    critique_schema = ResponseSchema(
        name="critique",
        description="""Very detailed and specific step-by-step instructions on how to improve Current chapter version in order to achieve the highest story quality.
        These instructions should describe step-by-step the recommendations to follow and/or the changes to make.
        Format: 
        1) ...
        2) ...
        3) ...
        """,
        validation=lambda x: isinstance(x, str)
        and x.count("\n") > 1,  # Must have multiple lines
        postprocess=lambda x: x.strip().split("\n"),  # Split into separate steps
    )

    rationale_schema = ResponseSchema(
        name="rationale",
        description="Rationale behind your critique.",
        validation=lambda x: isinstance(x, str)
        and len(x) > 20,  # Should be a substantial explanation
        postprocess=lambda x: x.strip(),  # Trim whitespace
    )

    plot_consistency_score_schema = ResponseSchema(
        name="plot_consistency_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Plot Consistency: Does the chapter maintain the integrity of the overall plot? Are there any inconsistencies or plot holes?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    character_development_score_schema = ResponseSchema(
        name="character_development_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Character Development: Are the characters in the chapter well-rounded and do they undergo meaningful changes or face challenges?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    engagement_score_schema = ResponseSchema(
        name="engagement_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Engagement: Is the chapter engaging from start to finish? Does it have a mixture of tension, release, and pacing?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    clarity_score_schema = ResponseSchema(
        name="clarity_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Clarity: Is the chapter easy to follow? Are events and character motivations clearly presented?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    detailing_score_schema = ResponseSchema(
        name="detailing_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Detailing: Does the chapter include sufficient details to make scenes vivid and memorable?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    language_quality_score_schema = ResponseSchema(
        name="language_quality_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Language Quality: Is the language varied, grammatically correct, and does it fit the tone of the story, the audience, etc.?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    dialogue_score_schema = ResponseSchema(
        name="dialogue_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Dialogue: Is the dialogue between characters natural, relevant, and does it serve the plot?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    emotional_impact_score_schema = ResponseSchema(
        name="emotional_impact_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Emotional Impact: Does the chapter evoke the desired emotional responses from the reader? (e.g., suspense, joy, sadness)""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    originality_score_schema = ResponseSchema(
        name="originality_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Originality: Is the chapter original, or does it offer new twists on familiar tropes?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    closure_score_schema = ResponseSchema(
        name="closure_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Closure: Does the chapter conclude in a way that is satisfying but also makes the reader want to move on to the next chapter?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    alignment_with_previous_chapter_score_schema = ResponseSchema(
        name="alignment_with_previous_chapter_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its Alignment with Previous Chapter: Does the chapter align well with events and tone set in the previous chapters?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    general_alignment_with_story_score_schema = ResponseSchema(
        name="general_alignment_with_story_score",
        description="""Float variable between 0.0 to 10.0 (a single float and NOT a ratio like 'x.x/10'), that scores the chapter in its General Alignment with Story: Does the chapter align well with with all story information availalbe (Profile of the target indivudual, Story featrures, Synopsis, etc.)?""",
        validation=lambda x: isinstance(x, float)
        and 0.0 <= x <= 10.0,  # Must be a float between 0.0 and 10.0
        postprocess=float,  # Ensure the value is a float
    )

    response_schemas = [
        critique_schema,
        # rationale_schema,
        plot_consistency_score_schema,
        character_development_score_schema,
        engagement_score_schema,
        clarity_score_schema,
        detailing_score_schema,
        language_quality_score_schema,
        dialogue_score_schema,
        emotional_impact_score_schema,
        originality_score_schema,
        closure_score_schema,
        alignment_with_previous_chapter_score_schema,
        general_alignment_with_story_score_schema,
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
    response_examples_prompt = load_prompt("critic/response_examples")
    system_message_template = load_prompt("critic/system_message")

    # Prepare the main prompt template
    main_prompt_template = PromptTemplate.from_template(
        system_message_template,
        partial_variables={
            "input_format_description": _get_input_format_description(),
            "response_format_instructions": _get_response_format_instructions_prompt_and_output_parser()[
                0
            ],
            "response_examples": response_examples_prompt,
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
        storys (str): A str containing storys.

    Returns:
        Tuple[SystemMessage, StructuredOutputParser]: A tuple containing a system message and a parser for structured output.
    """
    (
        system_message_prompt_template,
        output_parser,
    ) = _get_system_message_prompt_template_and_parser()
    system_message = system_message_prompt_template.format()
    return system_message, output_parser


def compose_message_for_LLM(
    story_blueprint: Dict[str, str],
    previous_chapter_content: str,
    input_info: Dict[str, str],
) -> HumanMessage:
    """
    Constructs a human-readable message based on the given input information.

    Args:
        input_info (Dict[str, str]): A dictionary containing input information.

    Returns:
        HumanMessage: The constructed human-readable message.
    """
    # Load the instructions template
    system_message_template = load_prompt("critic/input_format")
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(
        system_message_template
    )

    # Now, create the human message using the general info and the episodic_results
    human_message = human_message_prompt_template.format(
        chapter_number=input_info.get("chapter_number", 1),
        total_num_chapters=input_info.get("total_num_chapters", 1),
        current_chapter=input_info.get("chapter_generator_response_dict", {}).get(
            "content", " "
        ),
        story_blueprint=story_blueprint,
        previous_chapter=previous_chapter_content,
    )

    return human_message
