from datetime import date, datetime
from pathlib import Path

import pytest

from notebook_ocr.vault import PageEntry, build_markdown, write_notebook


def _pages():
    return [
        PageEntry(1, "IMG_4023.jpg", datetime(2024, 3, 14, 9, 12, 4), "# Day one\n\nnotes"),
        PageEntry(2, "IMG_4024.jpg", datetime(2024, 3, 14, 9, 12, 31), "more notes"),
    ]


@pytest.mark.unit
def test_front_matter_reports_notebook_and_page_count():
    md = build_markdown("field-notes-2024", Path("/photos"), _pages(), date(2026, 8, 7))

    assert "notebook: field-notes-2024" in md
    assert "pages: 2" in md
    assert "source_dir: /photos" in md
    assert "generated: 2026-08-07" in md


@pytest.mark.unit
def test_per_page_markers_include_filename_and_capture_time():
    md = build_markdown("nb", Path("/photos"), _pages(), date(2026, 8, 7))

    assert "<!-- page 001 (IMG_4023.jpg, 2024-03-14 09:12:04) -->" in md
    assert "<!-- page 002 (IMG_4024.jpg, 2024-03-14 09:12:31) -->" in md


@pytest.mark.unit
def test_transcription_text_is_preserved_verbatim():
    md = build_markdown("nb", Path("/photos"), _pages(), date(2026, 8, 7))

    assert "# Day one" in md
    assert "more notes" in md


@pytest.mark.unit
def test_write_notebook_creates_file_named_after_notebook(tmp_path):
    out = write_notebook(tmp_path / "vault", "my-notebook", "hello\n")

    assert out == tmp_path / "vault" / "my-notebook.md"
    assert out.read_text() == "hello\n"
