"""Discover input images and derive page order from capture time (filename-free)."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"})

# EXIF tag identifiers.
_EXIF_IFD_POINTER = 0x8769
_DATETIME_ORIGINAL = 36867


@dataclass(frozen=True)
class DiscoveredImage:
    """One input photo with its derived ordering key and content hash."""

    path: Path
    capture_time: datetime
    sha256: str


def read_capture_time(path: Path) -> datetime:
    """EXIF DateTimeOriginal if present, else the filesystem mtime (screenshots, stripped metadata)."""
    try:
        with Image.open(path) as img:
            raw = img.getexif().get_ifd(_EXIF_IFD_POINTER).get(_DATETIME_ORIGINAL)
        if raw:
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except (OSError, ValueError):
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def sha256_file(path: Path) -> str:
    """SHA-256 of the file bytes, streamed so large images don't load fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover(input_dir: Path) -> list[DiscoveredImage]:
    """Return image files sorted into page order: by capture time, then filename as a stable tiebreak."""
    paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images = [
        DiscoveredImage(path=path, capture_time=read_capture_time(path), sha256=sha256_file(path))
        for path in paths
    ]
    images.sort(key=lambda image: (image.capture_time, image.path.name))
    return images
