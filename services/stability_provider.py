"""
services/stability_provider.py

Text-to-image provider backed by the official Stability AI REST API
(v2beta stable-image endpoints). Uses `requests` directly since Stability's
lightweight REST surface doesn't require a heavyweight SDK dependency.
"""

from __future__ import annotations

import io

import requests
from PIL import Image

from config import settings
from models.request_model import GenerationRequest
from services.base_provider import BaseImageProvider
from services.openai_provider import ProviderError
from utils.logger import logger

_BASE_URL = "https://api.stability.ai/v2beta/stable-image/generate"

_ASPECT_RATIOS = {"1:1", "16:9", "9:16", "4:3", "3:2", "21:9"}


class StabilityProvider(BaseImageProvider):
    name = "Stability AI"
    available_models = ["stable-image-ultra", "stable-image-core", "sd3.5-large"]
    supported_advanced_params = {"cfg_scale", "seed", "inference_steps"}

    def is_configured(self) -> bool:
        return bool(settings.stability_api_key)

    def _endpoint_for_model(self, model: str) -> str:
        mapping = {
            "stable-image-ultra": f"{_BASE_URL}/ultra",
            "stable-image-core": f"{_BASE_URL}/core",
            "sd3.5-large": f"{_BASE_URL}/sd3",
        }
        return mapping.get(model, f"{_BASE_URL}/core")

    def generate(self, request: GenerationRequest, final_prompt: str) -> list[Image.Image]:
        model = request.model or "stable-image-core"
        endpoint = self._endpoint_for_model(model)
        aspect_ratio = request.aspect_ratio.value if request.aspect_ratio.value in _ASPECT_RATIOS else "1:1"

        headers = {
            "Authorization": f"Bearer {settings.stability_api_key}",
            "Accept": "image/*",
        }

        data = {
            "prompt": final_prompt,
            "output_format": "png",
            "aspect_ratio": aspect_ratio,
        }
        if request.negative_prompt.strip():
            data["negative_prompt"] = request.negative_prompt.strip()
        if request.seed is not None:
            data["seed"] = str(request.seed)
        if model == "sd3.5-large":
            data["cfg_scale"] = str(request.cfg_scale)
            data["model"] = "sd3.5-large"

        images: list[Image.Image] = []
        try:
            logger.info(
                "Stability AI request | model=%s aspect=%s n=%s",
                model,
                aspect_ratio,
                request.num_images,
            )
            for _ in range(request.num_images):
                response = requests.post(
                    endpoint,
                    headers=headers,
                    files={"none": ""},
                    data=data,
                    timeout=settings.request_timeout_seconds,
                )
                if response.status_code == 401:
                    raise ProviderError("Invalid Stability AI API key. Check your .env file.")
                if response.status_code == 429:
                    raise ProviderError("Stability AI rate limit reached. Please wait and try again.")
                if response.status_code != 200:
                    detail = response.text[:300]
                    raise ProviderError(f"Stability AI API error ({response.status_code}): {detail}")

                images.append(Image.open(io.BytesIO(response.content)).convert("RGB"))
        except requests.exceptions.ConnectionError as exc:
            raise ProviderError("Could not reach Stability AI — check your internet connection.") from exc
        except requests.exceptions.Timeout as exc:
            raise ProviderError("Stability AI request timed out. Please try again.") from exc

        if not images:
            raise ProviderError("Stability AI returned no images for this request.")

        return images
