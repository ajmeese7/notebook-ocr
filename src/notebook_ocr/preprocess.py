"""Local image cleanup before transcription: deskew, gentle auto-crop, contrast, downscale.

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

# Skew below this is not worth a resampling pass.
_MIN_SKEW_DEGREES = 0.5

# Claude downscales anything larger than this to fit its high-resolution vision tier,
# so sending more pixels costs bytes without adding detail the model can use. Capping
# here also keeps the base64 payload well under the API's 5 MB per-image limit — a
# straight-from-the-phone 12 MP page otherwise encodes to ~6 MB and is rejected.
MAX_LONG_EDGE_PX = 2576


def preprocess_image(path: Path) -> bytes:
    """Load an image (any supported format, incl. HEIC), clean it up, return PNG bytes."""
    with Image.open(path) as img:
        upright = ImageOps.exif_transpose(img)  # honor camera orientation before we drop EXIF
        gray = np.array(upright.convert("L"))

    gray = _deskew(gray)
    gray = _autocrop(gray)
    gray = _enhance_contrast(gray)
    gray = _downscale(gray)

    ok, buffer = cv2.imencode(".png", gray)
    if not ok:  # pragma: no cover - cv2 PNG encoding does not fail for valid arrays
        raise RuntimeError(f"failed to PNG-encode preprocessed image: {path}")
    return buffer.tobytes()


def estimate_skew(gray: np.ndarray) -> float:
    """Skew angle in degrees: positive means the page is rotated counter-clockwise."""
    inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(inverted > 0))
    if coords.size == 0:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    # Normalize into (-45, 45]: the rect's labelled edge is arbitrary, so 88 degrees of
    # "rotation" is really -2. OpenCV has reported this range as both [-90, 0] and
    # [0, 90) across versions; folding both ways keeps this version-independent.
    angle %= 90
    if angle > 45:
        angle -= 90
    return angle


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate the page upright by counter-rotating the estimated skew."""
    skew = estimate_skew(gray)
    if abs(skew) < _MIN_SKEW_DEGREES:
        return gray

    height, width = gray.shape
    center = (width / 2, height / 2)
    # Negate: to cancel a counter-clockwise skew we rotate clockwise by the same amount.
    matrix = cv2.getRotationMatrix2D(center, -skew, 1.0)
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


def _downscale(gray: np.ndarray) -> np.ndarray:
    """Shrink so the long edge fits MAX_LONG_EDGE_PX, preserving aspect ratio."""
    height, width = gray.shape
    long_edge = max(height, width)
    if long_edge <= MAX_LONG_EDGE_PX:
        return gray

    scale = MAX_LONG_EDGE_PX / long_edge
    new_size = (max(round(width * scale), 1), max(round(height * scale), 1))
    return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
