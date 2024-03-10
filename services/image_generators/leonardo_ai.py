import os
import json
import time
from omegaconf import DictConfig
from typing import List, Dict, Optional

import requests
import leonardoaisdk
from leonardoaisdk.models import operations, shared

from services.utils.log_utils import get_logger

# Get a logger instance for this module
logger = get_logger(__name__)


class LeonardoAiImageGenerator:
    """
    A class for handling image generation with Leonardo AI.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the LeonardoAiImageGenerator class with given configuration.

        Args:
            config (Dict): Configuration dictionary containing necessary parameters.
        """
        self.api_key = config.api_key
        self.sdk = leonardoaisdk.LeonardoAiSDK(
            security=shared.Security(bearer_auth=self.api_key)
        )
        self.config = config
        self.save_dir = ""

    def generate_images(
        self, image_prompts: List[Dict[str, str]], save_dir: str
    ) -> List[str]:
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
        image_filenames = [
            self._process_artifacts(self._request_image_generation(prompt), prompt)
            for prompt in image_prompts
        ]
        return [name for sublist in image_filenames for name in sublist]

    def _request_image_generation(self, prompt: str) -> Dict:
        """
        Generate responses based on a given prompt.

        Args:
            prompt (str): Image generation prompt.

        Returns:
            Dict: Generated responses from the API.
        """
        req = operations.CreateGenerationRequestBody(
            alchemy=False,  # self.config.alchemy,
            contrast_ratio=0.5,  # self.config.contrast_ratio,
            # control_net=False, # self.config.control_net,
            # control_net_type=shared.ControlnetType.DEPTH, #self.config.control_net_type,
            expanded_domain=False,  # self.config.expanded_domain,
            guidance_scale=8,  # self.config.guidance_scale,
            height=self.config.height,
            # high_contrast=True, #self.config.high_contrast,
            # high_resolution=False, #self.config.high_resolution,
            # image_prompt_weight=None, #self.config.image_prompt_weight,
            # image_prompts=None, #self.config.image_prompts,
            # init_generation_image_id=None, #self.config.init_generation_image_id,
            # init_image_id=None, #self.config.init_image_id,
            # init_strength=None, #self.config.init_strength,
            model_id="d69c8273-6b17-4a30-a13e-d6637ae1c644",  # self.config.model_id,
            # negative_prompt=None,
            # nsfw=self.config.nsfw,
            num_images=self.config.num_images,
            num_inference_steps=self.config.num_inference_steps,
            preset_style=shared.SdGenerationStyle.DYNAMIC,  # self.config.preset_style,
            prompt=prompt,
            prompt_magic=False,  # self.config.prompt_magic,
            # prompt_magic_version='v3', #self.config.prompt_magic_version,
            public=self.config.public,
            # scheduler=shared.SdGenerationSchedulers.DPM_SOLVER, #self.config.scheduler,
            # sd_version=shared.SdVersions.V1_5, # self.config.sd_version,
            # seed=self.config.seed,
            # tiling=False, # self.config.tiling,
            # unzoom=self.config.unzoom,
            # unzoom_amount=self.config.unzoom_amount,
            # upscale_ratio=self.config.upscale_ratio,
            # weighting=self.config.weighting,
            width=self.config.width,
        )
        return self.sdk.generation.create_generation(req)

    def _process_artifacts(self, res: Dict, prompt: str) -> List[str]:
        """
        Process and save the artifacts from the generated response.

        Args:
            res (Dict): Response from the generated request.
            prompt (str): The image prompt used.

        Returns:
            List[str]: List of saved image filenames.
        """
        image_filenames = []

        if res.create_generation_200_application_json_object:
            generation_id = (
                res.create_generation_200_application_json_object.sd_generation_job.generation_id
            )
            url = self.config.base_url + generation_id

            headers = {
                "accept": "application/json",
                "authorization": f"Bearer {self.api_key}",
            }

            for _ in range(self.config["max_retries"]):
                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    data = json.loads(response.text)
                    if (
                        data["generations_by_pk"] != "Null"
                        and data["generations_by_pk"]["status"] == "COMPLETE"
                    ):
                        image_url = data["generations_by_pk"]["generated_images"][0][
                            "url"
                        ]
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        image_filename = self._download_and_save_image(
                            image_url, image_name=f"{timestamp}_image.png"
                        )
                        image_filenames.append(image_filename)
                        break
                    else:
                        logger.warning(
                            f"Image is not ready to be downloaded yet. Retrying in {self.config['wait_time_secs']} seconds."
                        )
                        time.sleep(self.config["wait_time_secs"])

            else:  # else clause for for-loop, executed when loop completes normally (no break)
                logger.warning(
                    "Exceeded maximum retries. Image generation might have failed or is taking longer than expected."
                )
        else:
            logger.error(
                f"Request failed with status content {res.status_code}: {res.raw_response.reason}"
            )

        return image_filenames

    def _download_and_save_image(self, url: str, image_name: str) -> str:
        """
        Download an image and save it locally.

        Args:
            url (str): URL of the image.
            image_name (str): Desired local filename.

        Returns:
            str: Path to the saved image.
        """
        response = requests.get(url)
        response.raise_for_status()
        image_content = response.content
        image_path = os.path.join(self.save_dir, image_name)
        with open(image_path, "wb") as f:
            f.write(image_content)
        return image_path
