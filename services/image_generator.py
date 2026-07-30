"""
services/image_generator.py

Top-level orchestration service. This is the only class the UI layer
talks to: it validates the request, builds the structured prompt, calls
the selected provider (via dependency injection through the factory),
saves outputs to disk, and returns a fully-populated GenerationResult.
"""

from __future__ import annotations

import time

from models.request_model import GeneratedImage, GenerationRequest, GenerationResult
from prompts.prompt_builder import build_structured_prompt
from services.provider_factory import get_provider
from utils.downloader import save_image
from utils.image_utils import file_size_bytes, image_to_b64
from utils.logger import logger
from utils.validators import validate_num_images, validate_prompt, validate_provider_credentials


class ImageGenerationService:
    """Coordinates validation, prompt engineering, provider calls, and I/O."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.perf_counter()

        # --- Validation -----------------------------------------------------
        request.prompt = validate_prompt(request.prompt)
        validate_provider_credentials(request.provider)
        request.num_images = validate_num_images(request.num_images, request.provider)

        # --- Prompt engineering ---------------------------------------------
        final_prompt = build_structured_prompt(request)
        logger.info(
            "Generation requested | provider=%s style=%s images=%s prompt_len=%s",
            request.provider,
            request.style.value,
            request.num_images,
            len(request.prompt),
        )

        # --- Provider call (dependency-injected via the factory) -----------
        provider = get_provider(request.provider)
        pil_images = provider.generate(request, final_prompt)

        # --- Persist + build metadata ----------------------------------------
        generated_images: list[GeneratedImage] = []
        for index, pil_image in enumerate(pil_images):
            saved_path = save_image(pil_image, index)
            generated_images.append(
                GeneratedImage(
                    file_path=str(saved_path),
                    width=pil_image.width,
                    height=pil_image.height,
                    file_size_bytes=file_size_bytes(saved_path),
                    seed_used=request.seed,
                    b64_preview=image_to_b64(pil_image),
                )
            )

        elapsed = time.perf_counter() - start
        logger.info(
            "Generation complete | provider=%s images=%s duration=%.2fs",
            request.provider,
            len(generated_images),
            elapsed,
        )

        return GenerationResult(
            request=request,
            images=generated_images,
            provider_used=request.provider,
            model_used=request.model or provider.available_models[0],
            generation_seconds=round(elapsed, 2),
            final_prompt=final_prompt,
        )
