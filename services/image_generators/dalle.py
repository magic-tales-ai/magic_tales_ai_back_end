from openai import OpenAI

import requests
from omegaconf import DictConfig
import os
import logging

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class DALLEImageGenerator:
    """
    A class for handling conversations with OpenAI's DALLE model.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the OpenAIChatCompletion class with given configuration.

        Args:
            config (DictConfig): Configuration object containing necessary parameters.
        """
        self.client = OpenAI(api_key=config.api_key)
        self.num_images = config.num_images
        self.size = config.size
        self.save_dir = config.save_dir

    def generate_images(self, image_prompts):
        image_filenames = []
        for prompt in image_prompts:
            response = self.client.images.generate(
                prompt=prompt["caption"], n=self.num_images, size=self.size
            )
            for i, image_data in enumerate(response["data"]):
                image_url = image_data["url"]
                image_filename = self.download_and_save_image(
                    image_url,
                    save_dir=self.save_dir,
                    image_name=f"{prompt['caption']}_image_{i + 1}.png",
                )
                image_filenames.append(image_filename)
        return image_filenames

    @staticmethod
    def download_and_save_image(url, save_dir, image_name):
        response = requests.get(url)
        response.raise_for_status()
        image_content = response.content

        # Save the image
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image_path = os.path.join(save_dir, image_name)
        with open(image_path, "wb") as f:
            f.write(image_content)

        return image_path
