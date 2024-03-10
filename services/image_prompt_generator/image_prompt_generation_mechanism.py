import copy

from typing import Any, Dict, List

from langchain_community.chat_models import ChatOpenAI
from omegaconf import DictConfig

from services.prompts_constructors.image_prompt_generator import prompt_constructor

from services.image_prompt_generator.image_prompt_generator_LLM import (
    ImagePromptGeneratorLLM,
)
from services.image_prompt_generator.utils.response_utils import (
    initialize_image_prompt_response_dict,
)
from services.utils.log_utils import get_logger

# Initialize logger
logger = get_logger(__name__)


class ImagePromptGenerationMechanism:
    """
    The main class responsible for Magic Tales image prompt generation, using the Monte Carlo Tree Search algorithm.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the ImagePromptGenerationMechanism class.
        """
        self.config = config
        self.initialize_llms()
        self.top_level_creation_folder = None
        self.subfolders = {"image_prompts": "image_prompts"}

    def initialize_llms(self):
        """
        Initialize the LLMs for image prompt generation.
        """
        self.image_prompt_generator = ImagePromptGeneratorLLM(
            main_llm=ChatOpenAI(
                model_name=self.config.main_llm.model,
                temperature=self.config.main_llm.temperature,
                verbose=True,
                request_timeout=self.config.main_llm.request_timeout,
            ),
            parser_llm=ChatOpenAI(
                model_name=self.config.parser_llm.model,
                temperature=self.config.parser_llm.temperature,
                verbose=True,
                request_timeout=self.config.parser_llm.request_timeout,
            ),
            num_outputs=self.config.main_llm.num_responses,
            prompt_constructor=prompt_constructor,
        )

    def generate_image_prompts_for_all_chapters(
        self, chapters: List[Dict[str, str]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Generate image prompts for all chapters in the story.
        """
        logger.info("Generating image prompts for all chapters.")
        all_chapter_prompts = {}

        for i, chapter in enumerate(chapters):
            try:
                chapter_title = chapter.get("title", f"Chapter {i + 1}")
                chapter_content = chapter.get("content", "")
                logger.info(
                    f"Generating image prompts for Chapter {i + 1}: {chapter_title}"
                )

                image_prompt_data = self._generate_image_prompts_per_chapter(
                    i + 1, chapter_content
                )
                if image_prompt_data.get("image_prompt_generator_success"):
                    all_chapter_prompts[i] = {
                        "title": chapter_title,
                        "image_prompt_data": image_prompt_data,
                    }
                else:
                    logger.warning(
                        f"Failed to generate image prompts for {chapter_title}. Skipping."
                    )
            except Exception as e:
                logger.error(
                    f"Exception while generating image prompts for {chapter_title}: {e}",
                    exc_info=True,
                )

        return all_chapter_prompts

    def _generate_image_prompts_per_chapter(
        self, chapter_number: int, chapter_content: str
    ) -> Dict[str, Any]:
        """
        Generates image prompts for a given chapter content and annotates the chapter with image tags.
        """
        logger.info(
            "Generating chapter image prompts and placing image annotations inside the chapter."
        )

        for attempt in range(1, self.config.main_llm.max_retries + 1):
            try:
                (
                    image_prompt_responses,
                    image_prompt_generator_prompt_messages,
                ) = self.image_prompt_generator.generate_image_prompts(
                    chapter_number, chapter_content
                )
                (
                    image_prompt_generator_success,
                    image_prompt_response,
                ) = image_prompt_responses[0]

                if not image_prompt_generator_success:
                    raise RuntimeError("No image prompts responses generated")

                break
            except Exception as e:
                logger.error(
                    f"Failed to create image prompts for the chapter. Attempt {attempt}: {e}",
                    exc_info=True,
                )
                image_prompt_response = {
                    "annotated_chapter": None,
                    "image_prompts": None,
                }
                image_prompt_generator_success = False

        image_prompt_generator_response_dict = initialize_image_prompt_response_dict(
            image_prompt_generator_success=image_prompt_generator_success,
            image_prompt_generator_prompt_messages=image_prompt_generator_prompt_messages,
            image_prompt_response_content_dict=copy.deepcopy(image_prompt_response),
        )

        self._check_visualize_image_prompt_generator_response(
            image_prompt_generator_response_dict
        )

        return image_prompt_generator_response_dict

    def _check_visualize_image_prompt_generator_response(
        self, image_prompt_generator_response_dict: Dict[str, str]
    ) -> None:
        """
        Checks the configuration and visualizes the response from the image prompt generator.

        This method prints the response associated with a image prompt generator.
        The output is controlled by the `visualize_image_prompt_generator_response` configuration option.

        :param image_prompt_generator_response_dict: A dictionary containing information about the image prompt generator response.
        """
        if (
            self.config.output_artifacts.visualize_image_prompt_generator_response
            and image_prompt_generator_response_dict.get(
                "image_prompt_generator_success", False
            )
        ):
            chapter_content = image_prompt_generator_response_dict[
                "image_prompt_response_content_dict"
            ]["annotated_chapter"]
            image_prompts = image_prompt_generator_response_dict[
                "image_prompt_response_content_dict"
            ]["image_prompts"]

            print(f"\033[34mChapter:\033[0m")
            print(f"\033[34m{chapter_content}\033[0m")
            print(f"\033[34mImage Prompts:\033[0m")
            print(f"\033[34m{image_prompts}\033[0m")
