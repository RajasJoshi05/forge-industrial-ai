from PIL import Image

from industrial_ai.vision.image_validation import validate_image_file


def test_missing_image_is_rejected(tmp_path):
    errors = validate_image_file(tmp_path / "missing.png")

    assert len(errors) == 1
    assert "does not exist" in errors[0]


def test_valid_png_is_accepted(tmp_path):
    image_path = tmp_path / "valid.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    assert validate_image_file(image_path) == []


def test_corrupt_image_is_rejected(tmp_path):
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"this is not a valid image")

    errors = validate_image_file(image_path)

    assert len(errors) == 1
    assert "invalid or unreadable" in errors[0]
