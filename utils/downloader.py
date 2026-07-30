"""
utils/downloader.py

Handles persisting generated images to the local `outputs/` directory
using timestamped, collision-safe filenames, and provides byte payloads
for Streamlit's in-browser download button.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from config import settings
from utils.logger import logger


def build_filename(index: int = 0, extension: str = "png") -> str:
    """Build a timestamped, sortable filename, e.g. 2026_07_30_153522_01.png"""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    return f"{timestamp}_{index + 1:02d}.{extension}"


def save_image(image: Image.Image, index: int = 0) -> Path:
    """Save a Pillow image into outputs/ and return the saved path."""
    filename = build_filename(index)
    destination = settings.paths.outputs_dir / filename

    # Guard against same-second collisions when generating multiple images.
    counter = 1
    while destination.exists():
        destination = settings.paths.outputs_dir / f"{destination.stem}_{counter}{destination.suffix}"
        counter += 1

    image.save(destination, format="PNG")
    logger.info("Saved generated image to %s", destination)
    return destination


def image_bytes_for_download(path: Path) -> bytes:
    """Read saved image bytes back for a Streamlit download_button."""
    return path.read_bytes()
