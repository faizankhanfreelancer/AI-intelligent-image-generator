"""
prompts/prompt_builder.py

Turns raw user input into a professional, structured prompt before it is
ever sent to a provider. Also hosts the ready-made prompt library and the
random-prompt generator used by the "Surprise me" button.
"""

from __future__ import annotations

import random

from models.request_model import GenerationRequest

_STYLE_DESCRIPTORS = {
    "Photorealistic": "photorealistic, shot on a full-frame camera, natural lighting, tack-sharp detail",
    "Digital Art": "polished digital painting, vibrant color grading, trending on ArtStation",
    "Anime": "Japanese anime illustration, clean cel-shaded linework, expressive character design",
    "Fantasy": "epic fantasy illustration, dramatic atmosphere, painterly detail",
    "Cyberpunk": "neon-drenched cyberpunk aesthetic, high contrast, rain-slicked reflections",
    "Oil Painting": "traditional oil painting, visible brushwork, rich impasto texture",
    "Watercolor": "delicate watercolor illustration, soft bleeding pigments, textured paper",
    "Sketch": "graphite sketch, expressive linework, cross-hatched shading",
    "3D Render": "octane render, physically based materials, studio three-point lighting",
    "Comic": "comic book illustration, bold ink outlines, halftone shading",
    "Low Poly": "low-poly geometric art, faceted shading, minimal color palette",
    "Pixel Art": "16-bit pixel art, crisp dithering, retro game aesthetic",
    "Minimal": "minimalist composition, generous negative space, restrained palette",
    "Abstract": "abstract composition, expressive form, non-representational color fields",
    "Concept Art": "professional concept art, cinematic framing, production-ready detail",
}

_SYSTEM_PREAMBLE = "You are an expert AI image generation system. Generate an extremely detailed image."

_QUALITY_DESCRIPTORS = {
    "Standard": "clean, well-composed",
    "HD": "high definition, richly detailed",
    "Ultra": "ultra-detailed, hyper-realistic fidelity, flawless rendering",
}


def build_structured_prompt(request: GenerationRequest) -> str:
    """
    Compose the final prompt sent to the provider, following the
    structured template: subject, style, aspect ratio, quality,
    resolution, and negative prompt guidance.
    """
    style_descriptor = _STYLE_DESCRIPTORS.get(request.style.value, request.style.value)
    quality_descriptor = _QUALITY_DESCRIPTORS.get(request.quality.value, "")

    segments = [
        _SYSTEM_PREAMBLE,
        f"Subject: {request.prompt.strip()}",
        f"Style: {style_descriptor}",
        f"Aspect Ratio: {request.aspect_ratio.value}",
        f"Quality: {quality_descriptor} ({request.quality.value})",
        f"Resolution target: {request.resolution.value}",
        "Create cinematic lighting, high detail, professional composition, ultra realistic rendering.",
    ]

    if request.negative_prompt.strip():
        segments.append(f"Avoid the following at all costs: {request.negative_prompt.strip()}.")

    return " ".join(segments)


def enhance_prompt(raw_prompt: str) -> str:
    """
    A lightweight local 'prompt enhancer' — adds descriptive scaffolding
    to a short user idea without calling an external LLM. Useful when the
    user clicks 'Enhance Prompt'.
    """
    raw_prompt = raw_prompt.strip().rstrip(".")
    if not raw_prompt:
        return raw_prompt
    embellishments = [
        "intricate fine detail",
        "dramatic cinematic lighting",
        "balanced professional composition",
        "rich atmospheric depth",
    ]
    chosen = random.sample(embellishments, k=2)
    return f"{raw_prompt}, {', '.join(chosen)}."


PROMPT_LIBRARY: dict[str, str] = {
    "Fantasy Castle": "A towering fantasy castle perched on a cliffside, surrounded by mist and glowing lanterns at dusk.",
    "Cyberpunk City": "A futuristic cyberpunk city at night, flying cars, neon signage reflected on wet streets.",
    "Luxury Watch": "A macro product shot of a luxury mechanical watch on black velvet, dramatic studio lighting.",
    "Modern Office": "A bright, minimalist modern office interior with floor-to-ceiling windows and natural light.",
    "Space Station": "A vast orbital space station above Earth, sunlight glinting off solar panels, stars behind.",
    "Ancient Temple": "An overgrown ancient stone temple deep in a jungle, sunbeams piercing through the canopy.",
    "Portrait": "A close-up studio portrait of a person with striking eyes, soft rim lighting, shallow depth of field.",
    "Landscape": "A sweeping mountain landscape at golden hour, layered ridgelines fading into mist.",
    "Vehicle": "A sleek concept sports car on an empty highway at dawn, long exposure light trails.",
    "Architecture": "A striking piece of modern brutalist architecture, dramatic shadows, symmetrical composition.",
}

_RANDOM_SUBJECTS = [
    "a floating market above the clouds",
    "a lighthouse during a thunderstorm",
    "an underwater library filled with bioluminescent fish",
    "a robot gardener tending a rooftop greenhouse",
    "a desert caravan crossing dunes under twin moons",
    "a clockwork owl perched on a Victorian lamppost",
    "a hidden waterfall village carved into a cliff",
    "a samurai standing in a field of cherry blossoms",
    "an astronaut discovering ruins on a red planet",
    "a steampunk airship docking at a cloud city",
]

_RANDOM_MODIFIERS = [
    "bathed in golden hour light",
    "with dramatic storm clouds overhead",
    "rendered in a painterly style",
    "with intricate architectural detail",
    "under a sky full of stars",
    "with soft volumetric fog",
    "captured from a low dramatic angle",
]


def generate_random_prompt() -> str:
    """Produce a fresh, creative prompt for the 'Random Prompt' button."""
    subject = random.choice(_RANDOM_SUBJECTS)
    modifier = random.choice(_RANDOM_MODIFIERS)
    return f"A breathtaking scene of {subject}, {modifier}."
