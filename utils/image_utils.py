"""
utils/image_utils.py

Pillow-backed helpers: decoding base64/bytes into images, computing
dimensions and file sizes, and producing base64 previews for the UI.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def bytes_to_image(data: bytes) -> Image.Image:
    """Decode raw image bytes into a Pillow Image."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def b64_to_bytes(b64_string: str) -> bytes:
    """Decode a base64-encoded image string to raw bytes."""
    return base64.b64decode(b64_string)


def image_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a Pillow Image to a base64 string (for inline preview)."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def file_size_bytes(path: Path) -> int:
    """Return file size in bytes, 0 if the file does not exist."""
    return path.stat().st_size if path.exists() else 0


def human_readable_size(num_bytes: int) -> str:
    """Format a byte count as a friendly string, e.g. '1.4 MB'."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def aspect_ratio_to_dimensions(aspect_ratio: str, base: int = 1024) -> tuple[int, int]:
    """Convert an aspect ratio string like '16:9' into pixel dimensions."""
    ratios = {
        "1:1": (1024, 1024),
        "16:9": (1536, 864),
        "9:16": (864, 1536),
        "4:3": (1152, 864),
        "3:2": (1216, 810),
        "21:9": (1680, 720),
    }
    return ratios.get(aspect_ratio, (base, base))
