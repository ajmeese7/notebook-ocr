import io
import ipaddress
import json
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from notebook_ocr.cli import _DEFAULT_REVIEW_HOST
from notebook_ocr.config import Config
from notebook_ocr.discover import sha256_file
from notebook_ocr.review import _ALL_INTERFACES, browsable_url, create_app, lan_address
from notebook_ocr.state import State


def _write_page(directory, name, shade, captured_at):
    """A distinct grayscale page whose mtime fixes its position in page order."""
    path = directory / name
    Image.fromarray(np.full((120, 90), shade, np.uint8), mode="L").save(path)
    os.utime(path, (captured_at, captured_at))
    return path


@pytest.fixture
def notebook(tmp_path):
    """Two-page folder: page one is transcribed, page two never was."""
    photos = tmp_path / "photos" / "field-notes"
    photos.mkdir(parents=True)
    first = _write_page(photos, "b.jpg", 200, time.time() - 60)
    _write_page(photos, "a.jpg", 100, time.time())

    config = Config(
        input_dir=photos, vault_dir=tmp_path / "vault", state_file=tmp_path / "state.json"
    )
    state = State(config.state_file)
    state.put(sha256_file(first), "page one text", "claude-opus-5")
    state.save()
    return config


@pytest.fixture
def client(notebook):
    with TestClient(create_app(notebook)) as client:
        client.config = notebook
        yield client


@pytest.mark.unit
def test_notebook_lists_pages_in_capture_order(client):
    body = client.get("/api/notebook").json()

    assert body["notebook"] == "field-notes"
    assert [page["filename"] for page in body["pages"]] == ["b.jpg", "a.jpg"]
    assert [page["page_number"] for page in body["pages"]] == [1, 2]


@pytest.mark.unit
def test_transcribed_and_untranscribed_pages_are_distinguished(client):
    pages = client.get("/api/notebook").json()["pages"]

    assert pages[0]["status"] == "transcribed"
    assert pages[0]["text"] == "page one text"
    assert pages[1]["status"] == "missing"
    assert pages[1]["text"] == ""


@pytest.mark.unit
def test_page_image_endpoint_serves_a_decodable_png(client):
    sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

    response = client.get(f"/api/pages/{sha}/image")

    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).format == "PNG"


@pytest.mark.unit
def test_unknown_page_hash_returns_404(client):
    assert client.get("/api/pages/deadbeef/image").status_code == 404
    assert client.put("/api/pages/deadbeef", json={"text": "x"}).status_code == 404


@pytest.mark.unit
def test_saving_a_correction_persists_it_to_state(client):
    sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

    client.put(f"/api/pages/{sha}", json={"text": "corrected by hand"})

    entry = json.loads(client.config.state_file.read_text())["pages"][sha]
    assert entry["text"] == "corrected by hand"
    assert entry["edited_at"]


@pytest.mark.unit
def test_correction_preserves_the_original_transcription_metadata(client):
    """An edited page must stay distinguishable from raw model output."""
    sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

    response = client.put(f"/api/pages/{sha}", json={"text": "corrected"})

    assert response.json()["status"] == "edited"
    entry = json.loads(client.config.state_file.read_text())["pages"][sha]
    assert entry["model"] == "claude-opus-5"
    assert entry["transcribed_at"]


@pytest.mark.unit
def test_saving_a_correction_rebuilds_the_notebook_markdown(client):
    sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

    client.put(f"/api/pages/{sha}", json={"text": "corrected in the vault"})

    markdown = (client.config.vault_dir / "field-notes.md").read_text()
    assert "corrected in the vault" in markdown
    assert "<!-- page 001 (b.jpg" in markdown


@pytest.mark.unit
def test_rebuilt_markdown_marks_pages_that_were_never_transcribed(client):
    sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

    client.put(f"/api/pages/{sha}", json={"text": "corrected"})

    markdown = (client.config.vault_dir / "field-notes.md").read_text()
    assert "<!-- NOT TRANSCRIBED: a.jpg -->" in markdown
    assert "pages: 2" in markdown


@pytest.mark.unit
def test_correction_survives_a_reload_so_a_later_run_reuses_it(client):
    """The point of writing to state.json: `run` must not overwrite the correction."""
    sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

    client.put(f"/api/pages/{sha}", json={"text": "durable correction"})

    assert State(client.config.state_file).get_text(sha) == "durable correction"


@pytest.mark.unit
def test_empty_correction_is_rejected(client):
    sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

    assert client.put(f"/api/pages/{sha}", json={"text": ""}).status_code == 422


@pytest.mark.unit
def test_title_defaults_to_the_folder_name(client):
    body = client.get("/api/notebook").json()

    assert body["title"] == "field-notes"
    assert body["notebook"] == "field-notes"


@pytest.mark.unit
def test_renaming_persists_the_title_without_touching_page_identity(client):
    client.put("/api/notebook/title", json={"title": "Field Notes, 2024"})

    body = client.get("/api/notebook").json()
    assert body["title"] == "Field Notes, 2024"
    assert body["notebook"] == "field-notes"
    assert body["vault_file"].endswith("field-notes.md")


@pytest.mark.unit
def test_renaming_does_not_rename_the_vault_file(client):
    """A display title is cosmetic: renaming must never orphan the notebook markdown."""
    sha = client.get("/api/notebook").json()["pages"][0]["sha256"]
    client.put("/api/notebook/title", json={"title": "Renamed"})

    client.put(f"/api/pages/{sha}", json={"text": "corrected"})

    assert (client.config.vault_dir / "field-notes.md").exists()
    assert list(client.config.vault_dir.glob("*.md")) == [
        client.config.vault_dir / "field-notes.md"
    ]


@pytest.mark.unit
def test_blank_title_restores_the_folder_name(client):
    client.put("/api/notebook/title", json={"title": "Renamed"})

    body = client.put("/api/notebook/title", json={"title": "  "}).json()

    assert body["title"] == "field-notes"
    assert "notebooks" in json.loads(client.config.state_file.read_text())


@pytest.mark.unit
def test_overlong_title_is_rejected(client):
    response = client.put("/api/notebook/title", json={"title": "x" * 5000})

    assert response.status_code == 422


@pytest.mark.unit
def test_renaming_leaves_cached_transcriptions_intact(client):
    sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

    client.put("/api/notebook/title", json={"title": "Renamed"})

    assert State(client.config.state_file).get_text(sha) == "page one text"


@pytest.mark.unit
def test_index_serves_the_review_ui(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "<title>notebook-ocr review</title>" in response.text


@pytest.mark.unit
@pytest.mark.parametrize("wildcard", ["0.0.0.0", "::", ""])
def test_wildcard_bind_is_reported_as_a_loopback_url(wildcard):
    """A bind address is not a connectable one: opening http://0.0.0.0:8420 is not portable."""
    assert browsable_url(wildcard, 8420) == "http://127.0.0.1:8420"


@pytest.mark.unit
def test_explicit_host_is_reported_verbatim():
    assert browsable_url("192.168.1.50", 9000) == "http://192.168.1.50:9000"


@pytest.mark.unit
def test_default_review_host_is_reachable_from_the_network():
    """The UI is meant to be opened from a phone, so it must not bind loopback only."""
    assert _DEFAULT_REVIEW_HOST in _ALL_INTERFACES


@pytest.mark.unit
def test_lan_address_is_a_routable_ipv4_or_none():
    address = lan_address()

    if address is not None:
        assert ipaddress.IPv4Address(address)
        assert not ipaddress.IPv4Address(address).is_loopback
