"""
utils/validators.py

Input validation and lightweight prompt sanitization. Keeps user-supplied
text safe to embed into API payloads and prevents obviously invalid
requests (empty prompts, missing keys, unsupported combinations) from
reaching the network layer.
"""

from __future__ import annotations

import re

from config import settings

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MAX_PROMPT_LENGTH = 4000


class ValidationError(Exception):
    """Raised when a request fails validation before hitting a provider."""


def sanitize_text(text: str) -> str:
    """Strip control characters and excess whitespace from free text."""
    if not text:
        return ""
    cleaned = _CONTROL_CHARS.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:_MAX_PROMPT_LENGTH]


def validate_prompt(prompt: str) -> str:
    """Ensure the prompt is non-empty after sanitization."""
    cleaned = sanitize_text(prompt)
    if not cleaned:
        raise ValidationError("Please enter a prompt before generating an image.")
    if len(cleaned) < 3:
        raise ValidationError("Prompt is too short — please describe the image in more detail.")
    return cleaned


def validate_provider_credentials(provider_name: str) -> None:
    """Ensure the selected provider has an API key configured."""
    status = settings.provider_status()
    if provider_name not in status:
        raise ValidationError(f"Unknown provider: {provider_name}")
    if not status[provider_name]:
        raise ValidationError(
            f"No API key found for {provider_name}. "
            f"Add it to your .env file and restart the app."
        )


def validate_num_images(num_images: int, provider_name: str) -> int:
    """Clamp/validate image count per provider limits."""
    max_allowed = 8 if provider_name == "Stability AI" else 4
    if num_images < 1:
        raise ValidationError("Number of images must be at least 1.")
    if num_images > max_allowed:
        raise ValidationError(
            f"{provider_name} supports at most {max_allowed} images per request."
        )
    return num_images
