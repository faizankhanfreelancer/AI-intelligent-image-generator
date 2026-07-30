"""
services/provider_factory.py

Factory that maps a provider name (as chosen in the UI) to a concrete
BaseImageProvider instance. This is the single place that needs to
change when a new provider (Replicate, Hugging Face, Fal AI, Together AI,
Runware, ...) is added to the studio.
"""

from __future__ import annotations

from services.base_provider import BaseImageProvider
from services.openai_provider import OpenAIProvider
from services.stability_provider import StabilityProvider

_PROVIDER_REGISTRY: dict[str, type[BaseImageProvider]] = {
    "OpenAI": OpenAIProvider,
    "Stability AI": StabilityProvider,
}

_instances: dict[str, BaseImageProvider] = {}


def get_provider(name: str) -> BaseImageProvider:
    """Return a cached provider instance for the given provider name."""
    if name not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unknown provider '{name}'. Available: {list(_PROVIDER_REGISTRY)}")

    if name not in _instances:
        _instances[name] = _PROVIDER_REGISTRY[name]()

    return _instances[name]


def list_providers() -> list[str]:
    """Return all provider names registered with the factory."""
    return list(_PROVIDER_REGISTRY.keys())


def register_provider(name: str, provider_cls: type[BaseImageProvider]) -> None:
    """
    Register a new provider at runtime.

    Example (adding a future provider without touching the UI):
        from services.provider_factory import register_provider
        register_provider("Replicate", ReplicateProvider)
    """
    _PROVIDER_REGISTRY[name] = provider_cls
