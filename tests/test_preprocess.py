import io

import numpy as np
from PIL import Image

from notebook_ocr.preprocess import preprocess_image


def test_returns_decodable_png(tmp_path):
    path = tmp_path / "page.png"
    # A page with some dark "text" strokes on a light background.
    array = np.full((64, 48), 240, dtype=np.uint8)
    array[20:24, 8:40] = 20
    array[30:34, 8:30] = 20
    Image.fromarray(array, mode="L").save(path)

    png = preprocess_image(path)

    decoded = Image.open(io.BytesIO(png))
    assert decoded.format == "PNG"
    assert decoded.size[0] > 0 and decoded.size[1] > 0


def test_handles_blank_page_without_error(tmp_path):
    path = tmp_path / "blank.png"
    Image.new("L", (32, 32), 255).save(path)

    png = preprocess_image(path)

    assert Image.open(io.BytesIO(png)).format == "PNG"
