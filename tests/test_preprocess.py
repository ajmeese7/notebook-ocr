import base64
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from notebook_ocr.preprocess import (
    MAX_LONG_EDGE_PX,
    _deskew,
    estimate_skew,
    preprocess_image,
)

# Anthropic rejects a base64 image payload larger than this.
_API_IMAGE_LIMIT_BYTES = 5 * 1024 * 1024


def _ruled_page(skew_degrees: float, width: int = 600, height: int = 400) -> np.ndarray:
    """A light page with dark horizontal 'text lines', rotated counter-clockwise by skew."""
    page = np.full((height, width), 255, np.uint8)
    for y in range(80, height - 80, 40):
        page[y : y + 12, 60 : width - 60] = 0
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), skew_degrees, 1.0)
    return cv2.warpAffine(page, matrix, (width, height), borderValue=255)


@pytest.mark.unit
@pytest.mark.parametrize("skew", [1, 2, 5, 10, 20, -2, -5, -10, -20])
def test_deskew_reduces_skew_toward_zero(skew):
    """Regression: an inverted sign previously doubled the skew instead of correcting it."""
    skewed = _ruled_page(skew)

    corrected = _deskew(skewed)

    assert abs(estimate_skew(corrected)) < 0.6


@pytest.mark.unit
@pytest.mark.parametrize("skew", [2, -5, 10])
def test_deskew_never_increases_skew(skew):
    skewed = _ruled_page(skew)

    corrected = _deskew(skewed)

    assert abs(estimate_skew(corrected)) < abs(estimate_skew(skewed))


@pytest.mark.unit
def test_estimate_skew_sign_follows_rotation_direction():
    assert estimate_skew(_ruled_page(5)) > 0
    assert estimate_skew(_ruled_page(-5)) < 0


@pytest.mark.unit
def test_estimate_skew_of_blank_page_is_zero():
    assert estimate_skew(np.full((32, 32), 255, np.uint8)) == 0.0


@pytest.mark.unit
def test_large_photo_is_downscaled_to_model_limit(tmp_path):
    """Regression: a 12 MP phone photo previously encoded past the API's 5 MB limit."""
    path = tmp_path / "phone.png"
    Image.fromarray(np.full((3024, 4032), 200, np.uint8), mode="L").save(path)

    png = preprocess_image(path)

    decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert max(decoded.shape) <= MAX_LONG_EDGE_PX


@pytest.mark.unit
def test_realistic_photo_payload_stays_under_api_limit(tmp_path):
    path = tmp_path / "noisy.png"
    rng = np.random.default_rng(0)
    page = np.full((3024, 4032), 235, np.uint8)
    page += rng.integers(-12, 12, page.shape, dtype=np.int16).astype(np.uint8)
    for y in range(300, 2700, 90):
        page[y : y + 18, 400:3600] = 40
    Image.fromarray(page, mode="L").save(path)

    encoded = base64.standard_b64encode(preprocess_image(path))

    assert len(encoded) < _API_IMAGE_LIMIT_BYTES


@pytest.mark.unit
def test_small_image_is_not_upscaled(tmp_path):
    path = tmp_path / "small.png"
    Image.fromarray(np.full((100, 80), 200, np.uint8), mode="L").save(path)

    decoded = cv2.imdecode(
        np.frombuffer(preprocess_image(path), np.uint8), cv2.IMREAD_GRAYSCALE
    )

    assert max(decoded.shape) <= 100


@pytest.mark.unit
def test_returns_decodable_png(tmp_path):
    path = tmp_path / "page.png"
    Image.fromarray(_ruled_page(3), mode="L").save(path)

    png = preprocess_image(path)

    assert Image.open(io.BytesIO(png)).format == "PNG"


@pytest.mark.unit
def test_handles_blank_page_without_error(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("L", (32, 32), 255).save(path)

    assert Image.open(io.BytesIO(preprocess_image(path))).format == "PNG"
