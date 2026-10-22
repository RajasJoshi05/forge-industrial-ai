from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}


def validate_image_file(image_path: str | Path) -> list[str]:
    """Check whether a file exists and contains a readable supported image."""
    path = Path(image_path)
    errors: list[str] = []

    if not path.is_file():
        return [f"Image file does not exist: {path}"]

    try:
        with Image.open(path) as image:
            if image.format not in SUPPORTED_IMAGE_FORMATS:
                errors.append(f"Unsupported image format: {image.format}")
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        errors.append(f"Image file is invalid or unreadable: {path}")

    return errors
