import copy

from typing import Any, Dict, Union


from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM

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
        self.top_level_creation_folder = None
        self.subfolders = {"image_prompts": "image_prompts"}

        self.initialize_llms(self.config)

    def initialize_llms(self, config: DictConfig):
        """
        Initialize the LLMs for image prompt generation.
        """
        main_llm = self._load_llm(config)        
        parser_llm = self._load_llm(config)

        self.image_prompt_generator = ImagePromptGeneratorLLM(
            main_llm=main_llm,
            parser_llm=parser_llm,
            num_outputs=self.config.num_responses,
            prompt_constructor=prompt_constructor,
        )

    def _load_llm(self, llm_config: DictConfig) -> Union[ChatOpenAI, ChatAnthropic, OllamaLLM]:
        """
        Load an LLM model based on the provided configuration.

        Args:
            llm_config (DictConfig): The configuration for the LLM model.

        Returns:
            Union[ChatOpenAI, ChatAnthropic, OllamaLLM]: The loaded LLM model.
        """
        model_type = llm_config.model_type
        model_name = llm_config.model
        temperature = llm_config.temperature
        max_tokens = llm_config.max_tokens
        request_timeout = llm_config.request_timeout

        if model_type == "openai":
            return ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens, request_timeout=request_timeout)
        elif model_type == "anthropic":
            return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens, request_timeout=request_timeout)
        elif model_type == "ollama":
            return OllamaLLM(model=model_name, temperature=temperature, max_tokens=max_tokens, request_timeout=request_timeout)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _generate_single_image_prompt(self, prompt: str) -> Dict[str, Any]:
        """Generate a single image prompt."""
        try:
            image_prompt_responses, image_prompt_generator_prompt_messages = (
                self.image_prompt_generator.generate_single_image_prompt(prompt)
            )
            image_prompt_generator_success, image_prompt_response = (
                image_prompt_responses[0]
            )

            if not image_prompt_generator_success:
                raise RuntimeError("No image prompt response generated")

            return {
                "image_prompt_generator_success": True,
                "image_prompt_generator_prompt_messages": image_prompt_generator_prompt_messages,
                "image_prompt_response_content_dict": {
                    "annotated_chapter": prompt,  # We use the original prompt as there's no chapter to annotate
                    "image_prompts": [
                        image_prompt_response
                    ],  # Wrap in a list to maintain consistency with chapter prompts
                },
            }
        except Exception as e:
            logger.error(
                f"Failed to create image prompt for the cover. Error: {e}",
                exc_info=True,
            )
            return {"image_prompt_generator_success": False}

    def _generate_image_prompts_per_chapter(
        self, chapter_number: int, chapter_content: str, is_cover: bool = False
    ) -> Dict[str, Any]:
        """
        Generates image prompts for a given chapter content and annotates the chapter with image tags.
        """
        log_message = (
            "Generating cover image prompt."
            if is_cover
            else "Generating chapter image prompts and placing image annotations inside the chapter."
        )
        logger.info(log_message)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                (
                    image_prompt_responses,
                    image_prompt_generator_prompt_messages,
                ) = self.image_prompt_generator.generate_image_prompts(
                    chapter_number, chapter_content, is_cover
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
            image_prompt_generator_response_dict, is_cover
        )

        return image_prompt_generator_response_dict

    def _check_visualize_image_prompt_generator_response(
        self, image_prompt_generator_response_dict: Dict[str, str], is_cover: bool
    ) -> None:
        """
        Checks the configuration and visualizes the response from the image prompt generator.

        Args:
            image_prompt_generator_response_dict (Dict[str, str]): A dictionary containing information about the image prompt generator response.
            is_cover (bool): Whether this is for the cover image.
        """
        if (
            self.config.visualize_image_prompt_generator_response
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

            logger.info(f"\033[34m{'Cover' if is_cover else 'Chapter'}:\033[0m")
            logger.info(f"\033[34m{chapter_content}\033[0m")
            logger.info(f"\033[34mImage Prompts:\033[0m")
            logger.info(f"\033[34m{image_prompts}\033[0m")
