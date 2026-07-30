"""
services/openai_provider.py

Text-to-image provider backed by the official OpenAI SDK
(GPT Image / DALL·E models via `client.images.generate`).
"""

from __future__ import annotations

import base64
import io

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from PIL import Image

from config import settings
from models.request_model import GenerationRequest
from services.base_provider import BaseImageProvider
from utils.logger import logger

# Sizes officially supported by OpenAI's image models.
_SUPPORTED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}


class ProviderError(Exception):
    """Raised when a provider call fails in a user-facing way."""


class OpenAIProvider(BaseImageProvider):
    name = "OpenAI"
    available_models = ["gpt-image-1", "dall-e-3", "dall-e-2"]
    supported_advanced_params = {"quality"}

    def __init__(self) -> None:
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.request_timeout_seconds,
            )
        return self._client

    def is_configured(self) -> bool:
        return bool(settings.openai_api_key)

    def _closest_supported_size(self, resolution: str) -> str:
        if resolution in _SUPPORTED_SIZES:
            return resolution
        # Map unsupported sizes (e.g. 512x512, 2048x2048) to the nearest option.
        width_str, height_str = resolution.split("x")
        width, height = int(width_str), int(height_str)
        if width == height:
            return "1024x1024"
        return "1024x1536" if height > width else "1536x1024"

    def generate(self, request: GenerationRequest, final_prompt: str) -> list[Image.Image]:
        client = self._get_client()
        size = self._closest_supported_size(request.resolution.value)
        model = request.model or "gpt-image-1"

        # dall-e-2 does not support the "quality" or multi-size parameter set
        # that gpt-image-1 / dall-e-3 support, so branch accordingly.
        payload: dict = {
            "model": model,
            "prompt": final_prompt,
            "n": request.num_images if model != "dall-e-3" else 1,
            "size": size,
        }

        if model == "gpt-image-1":
            quality_map = {"Standard": "medium", "HD": "high", "Ultra": "high"}
            payload["quality"] = quality_map.get(request.quality.value, "medium")
        elif model == "dall-e-3":
            payload["quality"] = "hd" if request.quality.value in ("HD", "Ultra") else "standard"

        try:
            logger.info(
                "OpenAI request | model=%s size=%s n=%s", model, size, payload["n"]
            )
            response = client.images.generate(**payload)
        except RateLimitError as exc:
            raise ProviderError("OpenAI rate limit reached. Please wait a moment and try again.") from exc
        except APIConnectionError as exc:
            raise ProviderError("Could not reach OpenAI — check your internet connection.") from exc
        except APIStatusError as exc:
            if exc.status_code == 401:
                raise ProviderError("Invalid OpenAI API key. Check your .env file.") from exc
            raise ProviderError(f"OpenAI API error ({exc.status_code}): {exc.message}") from exc
        except Exception as exc:  # noqa: BLE001 - surface as a friendly provider error
            raise ProviderError(f"Unexpected OpenAI error: {exc}") from exc

        images: list[Image.Image] = []
        for item in response.data:
            if getattr(item, "b64_json", None):
                raw = base64.b64decode(item.b64_json)
            elif getattr(item, "url", None):
                import requests

                raw = requests.get(item.url, timeout=settings.request_timeout_seconds).content
            else:
                continue
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))

        # dall-e-3 only returns 1 image per call — replicate the call if more were requested.
        if model == "dall-e-3" and request.num_images > 1:
            for _ in range(request.num_images - 1):
                extra = client.images.generate(**payload)
                for item in extra.data:
                    if getattr(item, "b64_json", None):
                        raw = base64.b64decode(item.b64_json)
                        images.append(Image.open(io.BytesIO(raw)).convert("RGB"))

        if not images:
            raise ProviderError("OpenAI returned no images for this request.")

        return images
