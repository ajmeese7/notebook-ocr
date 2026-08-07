"""Local image cleanup before transcription: deskew, gentle auto-crop, contrast.

Kept deliberately simple. The vision model is robust to imperfect input, so the goal
is to remove obvious skew and boost legibility, not to perfectly rectify every page.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Only crop when a clear page border is detected: the content bounding box must cover
# at least this fraction of the frame. Otherwise cropping risks cutting into the page.
_MIN_CONTENT_FRACTION = 0.5
_CROP_MARGIN_PX = 12


def preprocess_image(path: Path) -> bytes:
    """Load an image (any supported format, incl. HEIC), clean it up, return PNG bytes."""
    with Image.open(path) as img:
        upright = ImageOps.exif_transpose(img)  # honor camera orientation before we drop EXIF
        gray = np.array(upright.convert("L"))

    gray = _deskew(gray)
    gray = _autocrop(gray)
    gray = _enhance_contrast(gray)

    ok, buffer = cv2.imencode(".png", gray)
    if not ok:  # pragma: no cover - cv2 PNG encoding does not fail for valid arrays
        raise RuntimeError(f"failed to PNG-encode preprocessed image: {path}")
    return buffer.tobytes()


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate the dominant text angle and rotate the page upright."""
    inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(inverted > 0))
    if coords.size == 0:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    if abs(angle) < 0.5:  # nothing meaningful to correct
        return gray

    height, width = gray.shape
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _autocrop(gray: np.ndarray) -> np.ndarray:
    """Crop away a uniform border when a clear page rectangle is present; otherwise leave as-is."""
    inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(inverted)
    if coords is None:
        return gray

    x, y, w, h = cv2.boundingRect(coords)
    height, width = gray.shape
    if (w * h) < (_MIN_CONTENT_FRACTION * width * height):
        return gray  # no confident border to trim

    x0 = max(x - _CROP_MARGIN_PX, 0)
    y0 = max(y - _CROP_MARGIN_PX, 0)
    x1 = min(x + w + _CROP_MARGIN_PX, width)
    y1 = min(y + h + _CROP_MARGIN_PX, height)
    return gray[y0:y1, x0:x1]


def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE: boost local contrast so faint pencil strokes separate from the page."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
