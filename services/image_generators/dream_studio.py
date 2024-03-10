import os
import io
import warnings
import logging
from omegaconf import DictConfig
from PIL import Image
from typing import List, Optional, Union, Dict
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class DreamStudioImageGenerator:
    """
    A class for handling image generation with DreamStudio's model.
    """

    def __init__(self, config: DictConfig) -> None:
        """
        Initialize the DreamStudioImageGenerator class with given configuration.

        Args:
            config (Union[Dict, object]): Configuration object or dict containing necessary parameters.
        """
        # Set up API Key and host from the configuration
        os.environ["STABILITY_HOST"] = config.host
        os.environ["STABILITY_KEY"] = config.api_key

        # Set up our connection to the API.
        self.stability_api = client.StabilityInference(
            key=os.environ["STABILITY_KEY"],
            verbose=True,
            engine=config.engine,
        )

        # Class variables from configuration
        self.steps = config.steps
        self.config_scale = config.config_scale
        self.width = config.width
        self.height = config.height
        self.samples = config.samples
        self.sampler = config.sampler
        self.save_dir = config.save_dir

    def generate_images(
        self, image_prompts: List[str], save_dir: Optional[str] = None
    ) -> List[str]:
        """
        Generate images based on provided prompts.

        Args:
            image_prompts (List[str]): List of image prompts.
            save_dir (str, optional): Directory to save generated images. Defaults to the class initialized save_dir.

        Returns:
            List[str]: List of filenames for the generated images.
        """
        target_dir = save_dir if save_dir else self.save_dir
        os.makedirs(target_dir, exist_ok=True)

        image_filenames = [
            filename
            for prompt in image_prompts
            for filename in self._process_artifacts(
                self._generate_prompt_responses(prompt).artifacts
            )
        ]

        return image_filenames

    def _generate_prompt_responses(self, prompt: str) -> generation:
        """
        Generate responses based on a given prompt.

        Args:
            prompt (str): Image generation prompt.

        Returns:
            generation: Generated responses from the stability API.
        """
        return self.stability_api.generate(
            prompt=prompt,
            steps=self.steps,
            config_scale=self.config_scale,
            width=self.width,
            height=self.height,
            samples=self.samples,
            sampler=self.sampler,
        )

    def _process_artifacts(self, artifacts: List) -> List[str]:
        """
        Process and save the artifacts from the generated response.

        Args:
            artifacts (List): List of artifacts from the generated response.

        Returns:
            List[str]: List of saved image filenames.
        """
        image_filenames = []

        for artifact in artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again."
                )
                continue

            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                image_filename = f"{artifact.seed}.png"
                img.save(os.path.join(self.save_dir, image_filename))
                image_filenames.append(image_filename)

        return image_filenames
