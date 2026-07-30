"""
services/base_provider.py

Defines the abstract contract every image-generation provider must
implement. Adding a new provider (Replicate, Hugging Face, Fal AI,
Together AI, Runware, ...) means writing one class that satisfies this
interface — the UI and orchestration layer never need to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from models.request_model import GenerationRequest


class BaseImageProvider(ABC):
    """Abstract base class for all text-to-image providers."""

    #: Human-readable provider name shown in the UI.
    name: str = "base"

    #: Model identifiers this provider exposes to the UI.
    available_models: list[str] = []

    #: Advanced parameters this provider actually honors.
    supported_advanced_params: set[str] = set()

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if credentials are present for this provider."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: GenerationRequest, final_prompt: str) -> list[Image.Image]:
        """
        Generate one or more images for the given request.

        Args:
            request: The validated, structured generation request.
            final_prompt: The fully-composed prompt text to send.

        Returns:
            A list of Pillow Image objects.
        """
        raise NotImplementedError
