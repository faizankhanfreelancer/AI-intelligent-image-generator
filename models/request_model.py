"""
models/request_model.py

Pydantic models describing the shape of a generation request, a single
generated image, and a generation result. Using Pydantic gives us
validation, type safety, and easy serialization for history persistence.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class StyleOption(str, Enum):
    PHOTOREALISTIC = "Photorealistic"
    DIGITAL_ART = "Digital Art"
    ANIME = "Anime"
    FANTASY = "Fantasy"
    CYBERPUNK = "Cyberpunk"
    OIL_PAINTING = "Oil Painting"
    WATERCOLOR = "Watercolor"
    SKETCH = "Sketch"
    RENDER_3D = "3D Render"
    COMIC = "Comic"
    LOW_POLY = "Low Poly"
    PIXEL_ART = "Pixel Art"
    MINIMAL = "Minimal"
    ABSTRACT = "Abstract"
    CONCEPT_ART = "Concept Art"


class QualityOption(str, Enum):
    STANDARD = "Standard"
    HD = "HD"
    ULTRA = "Ultra"


class AspectRatio(str, Enum):
    SQUARE = "1:1"
    WIDE = "16:9"
    TALL = "9:16"
    STANDARD_4_3 = "4:3"
    STANDARD_3_2 = "3:2"
    CINEMA = "21:9"


class Resolution(str, Enum):
    R512 = "512x512"
    R768 = "768x768"
    R1024 = "1024x1024"
    R1024_1536 = "1024x1536"
    R1536_1024 = "1536x1024"
    R2048 = "2048x2048"


class SafetyLevel(str, Enum):
    STRICT = "Strict"
    STANDARD = "Standard"
    RELAXED = "Relaxed"


class GenerationRequest(BaseModel):
    """A fully-specified request to generate one or more images."""

    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=2000)
    provider: str = Field(default="OpenAI")
    model: str = Field(default="")
    style: StyleOption = StyleOption.PHOTOREALISTIC
    resolution: Resolution = Resolution.R1024
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    quality: QualityOption = QualityOption.STANDARD
    num_images: int = Field(default=1, ge=1, le=8)
    seed: Optional[int] = Field(default=None, ge=0, le=2_147_483_647)
    safety_level: SafetyLevel = SafetyLevel.STANDARD

    # Advanced / provider-specific parameters (only honored where supported)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=35.0)
    inference_steps: int = Field(default=30, ge=10, le=150)
    scheduler: str = Field(default="DDIM")

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Prompt cannot be empty or whitespace only.")
        return cleaned


class GeneratedImage(BaseModel):
    """A single generated image and its metadata."""

    file_path: str
    width: int
    height: int
    file_size_bytes: int
    seed_used: Optional[int] = None
    b64_preview: Optional[str] = None


class GenerationResult(BaseModel):
    """The outcome of a generation call: images plus timing/metadata."""

    request: GenerationRequest
    images: list[GeneratedImage] = Field(default_factory=list)
    provider_used: str
    model_used: str
    generation_seconds: float
    created_at: datetime = Field(default_factory=datetime.now)
    final_prompt: str = ""

    class Config:
        arbitrary_types_allowed = True
