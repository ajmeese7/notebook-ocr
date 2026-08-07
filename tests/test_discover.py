import os
from datetime import datetime

from PIL import Image

from notebook_ocr.discover import IMAGE_EXTENSIONS, discover, read_capture_time, sha256_file


def _write_png(path, color=(255, 255, 255)):
    Image.new("RGB", (8, 8), color).save(path)


def test_orders_by_mtime_when_no_exif(tmp_path):
    first = tmp_path / "b.png"
    second = tmp_path / "a.png"
    _write_png(first)
    _write_png(second)
    os.utime(first, (1000, 1000))
    os.utime(second, (2000, 2000))

    ordered = [image.path.name for image in discover(tmp_path)]

    assert ordered == ["b.png", "a.png"]


def test_identical_timestamp_breaks_tie_by_filename(tmp_path):
    later = tmp_path / "z.png"
    earlier = tmp_path / "a.png"
    _write_png(later)
    _write_png(earlier)
    os.utime(later, (5000, 5000))
    os.utime(earlier, (5000, 5000))

    ordered = [image.path.name for image in discover(tmp_path)]

    assert ordered == ["a.png", "z.png"]


def test_ignores_non_image_files(tmp_path):
    _write_png(tmp_path / "page.png")
    (tmp_path / "notes.txt").write_text("ignore me")

    names = [image.path.name for image in discover(tmp_path)]

    assert names == ["page.png"]


def test_reads_exif_datetime_original(tmp_path):
    path = tmp_path / "shot.jpg"
    img = Image.new("RGB", (8, 8), (0, 0, 0))
    exif = img.getexif()
    exif.get_ifd(0x8769)[36867] = "2024:03:14 09:12:04"
    img.save(path, exif=exif)

    assert read_capture_time(path) == datetime(2024, 3, 14, 9, 12, 4)


def test_sha256_matches_content(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_png(a, color=(0, 0, 0))
    _write_png(b, color=(0, 0, 0))

    assert sha256_file(a) == sha256_file(b)


def test_heic_extension_recognized():
    assert ".heic" in IMAGE_EXTENSIONS
