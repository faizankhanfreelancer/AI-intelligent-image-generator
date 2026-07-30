"""
config.py

Centralized application configuration.

Loads environment variables via python-dotenv and exposes a single
`settings` object that the rest of the application imports from.
Never hardcode secrets here — everything sensitive comes from `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root, regardless of current working directory.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass(frozen=True)
class Paths:
    """Filesystem locations used throughout the app."""

    base_dir: Path = BASE_DIR
    outputs_dir: Path = BASE_DIR / "outputs"
    history_dir: Path = BASE_DIR / "history"
    assets_dir: Path = BASE_DIR / "assets"
    logs_dir: Path = BASE_DIR / "history" / "logs"

    def ensure(self) -> None:
        for directory in (
            self.outputs_dir,
            self.history_dir,
            self.assets_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Application-wide settings, populated from environment variables."""

    app_name: str = "Multimodal Image Generation Studio"
    app_version: str = "1.0.0"

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip())
    stability_api_key: str = field(default_factory=lambda: os.getenv("STABILITY_API_KEY", "").strip())

    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    default_provider: str = os.getenv("DEFAULT_PROVIDER", "OpenAI")

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    paths: Paths = field(default_factory=Paths)

    def provider_status(self) -> dict:
        """Return which providers currently have credentials configured."""
        return {
            "OpenAI": bool(self.openai_api_key),
            "Stability AI": bool(self.stability_api_key),
        }


settings = Settings()
settings.paths.ensure()
