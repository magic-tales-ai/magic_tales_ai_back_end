import os
import requests
from typing import List, Dict
from omegaconf import DictConfig
from openai import OpenAI

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class DALLE3ImageGenerator:
    """
    A class for handling image generation using OpenAI's DALL-E 3 model.

    Attributes:
        client (OpenAI): An instance of OpenAI client for API interaction.
        num_images (int): Number of images to generate for each prompt.
        size (str): Size of the images to generate (e.g., "1024x1024").
        quality (str): Quality of the images ("normal" or "hd").
        save_dir (str): Directory where generated images will be saved.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the DALLE3ImageGenerator class with the given configuration.

        Args:
            config (DictConfig): Configuration object containing necessary parameters.
        """
        self.client = OpenAI()
        self.num_images = config.num_images
        self.size = config.size
        self.quality = config.quality
        self.save_dir = ""

    def generate_images(self, image_prompts: List[str], save_dir: str) -> List[str]:
        """
        Generates images using DALL-E 3 based on provided prompts.

        Each prompt in the 'image_prompts' list is a dictionary containing a caption that guides the image generation process.
        This method sends the prompts to the DALL-E 3 API and saves the generated images to the specified directory.
        Each image is named uniquely to avoid overwriting existing files.

        Args:
            image_prompts (List[Dict[str, str]]): A list of dictionaries where each dictionary contains a prompt caption
                                                  under a key such as 'caption'. Example: [{"caption": "A robot playing the piano"}]
            save_dir (str): Directory where generated images will be saved.

        Returns:
            List[str]: A list of file paths where the generated images are saved. These file paths are relative to
                       the 'save_dir' specified in the class initializer.

        Raises:
            Exception: If there's an error in generating or saving images.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        image_filenames = []
        for prompt in image_prompts:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=self.size,
                quality=self.quality,
                n=self.num_images,
            )
            for i, image_data in enumerate(response.data):
                image_url = image_data.url
                image_filename = self.download_and_save_image(
                    image_url,
                    save_dir=self.save_dir,
                    image_name=f"image_{i + 1}.png",
                )
                image_filenames.append(image_filename)
        return image_filenames

    @staticmethod
    def download_and_save_image(url, save_dir, image_name):
        """
        Downloads an image from a URL and saves it to a specified directory.

        Args:
            url (str): URL of the image to download.
            save_dir (str): Directory to save the downloaded image.
            image_name (str): Name to be given to the saved image file.

        Returns:
            str: Path to the saved image file.
        """
        response = requests.get(url)
        response.raise_for_status()
        image_content = response.content

        # Ensure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the image
        image_path = os.path.join(save_dir, image_name)
        with open(image_path, "wb") as f:
            f.write(image_content)

        logger.info(f"Image saved at {image_path}")
        return image_path
