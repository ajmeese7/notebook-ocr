import io
import ipaddress
import json
import os
import time

import anthropic
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from notebook_ocr.cli import _DEFAULT_REVIEW_HOST
from notebook_ocr.config import Config
from notebook_ocr.discover import sha256_file
from notebook_ocr.review import _ALL_INTERFACES, browsable_url, create_app, lan_address
from notebook_ocr.state import State
from notebook_ocr.transcribe import (
    CredentialsMissing,
    TranscriptionError,
    TranscriptionRefused,
)


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


def _make_client(config, transcribe):
    """A review client whose re-transcribe endpoint runs `transcribe` instead of the API.

    Substituting the transcription call, not the HTTP client underneath it: everything the
    endpoint is responsible for (preprocessing, cache writes, markdown rebuild, error
    mapping) still runs for real. The alternative is billing a live call per test.
    """
    client = TestClient(create_app(config, transcribe=transcribe))
    client.config = config
    return client


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
def test_refused_page_is_distinguished_from_one_never_attempted(client):
    """A refusal will not clear on a re-run; an untried page will. They are not the same call to action."""
    untried = client.get("/api/notebook").json()["pages"][1]
    state = State(client.config.state_file)
    state.put_failure(untried["sha256"], "refused", "TranscriptionRefused: model refused page")
    state.save()

    page = client.get("/api/notebook").json()["pages"][1]

    assert page["status"] == "failed"
    assert page["failure"] == "refused"


@pytest.mark.unit
def test_page_no_run_has_reached_reports_no_failure(client):
    page = client.get("/api/notebook").json()["pages"][1]

    assert page["status"] == "missing"
    assert page["failure"] is None


@pytest.mark.unit
def test_rebuilt_markdown_keeps_the_failure_reason(client):
    """Rebuilding after a correction must not downgrade "the model refused" to "not transcribed"."""
    pages = client.get("/api/notebook").json()["pages"]
    state = State(client.config.state_file)
    state.put_failure(pages[1]["sha256"], "refused", "TranscriptionRefused: model refused page")
    state.save()

    client.put(f"/api/pages/{pages[0]['sha256']}", json={"text": "corrected"})

    markdown = (client.config.vault_dir / "field-notes.md").read_text()
    assert "<!-- TRANSCRIPTION FAILED: TranscriptionRefused: model refused page -->" in markdown
    assert "NOT TRANSCRIBED" not in markdown


@pytest.mark.unit
def test_correcting_a_refused_page_clears_the_flag(client):
    sha = client.get("/api/notebook").json()["pages"][1]["sha256"]
    state = State(client.config.state_file)
    state.put_failure(sha, "refused", "TranscriptionRefused: model refused page")
    state.save()

    page = client.put(f"/api/pages/{sha}", json={"text": "typed out by hand"}).json()

    assert page["status"] == "edited"
    assert page["failure"] is None


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
def test_notebook_names_the_model_a_retranscribe_would_bill(client):
    """The confirmation has to name the charge, so the UI must be told what it is."""
    assert client.get("/api/notebook").json()["model"] == "claude-opus-5"


@pytest.mark.unit
def test_retranscribe_replaces_the_cached_text(notebook):
    with _make_client(notebook, lambda png: ("fresh output", "claude-opus-5")) as client:
        sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

        page = client.post(f"/api/pages/{sha}/transcribe").json()

    assert page["text"] == "fresh output"
    assert page["status"] == "transcribed"
    assert State(notebook.state_file).get_text(sha) == "fresh output"


@pytest.mark.unit
def test_retranscribe_sends_the_preprocessed_image(notebook):
    """The model must see what `run` would send, not the raw photo off the camera."""
    seen = []

    with _make_client(notebook, lambda png: (seen.append(png), ("text", "m"))[1]) as client:
        sha = client.get("/api/notebook").json()["pages"][0]["sha256"]
        client.post(f"/api/pages/{sha}/transcribe")

    assert Image.open(io.BytesIO(seen[0])).format == "PNG"


@pytest.mark.unit
def test_retranscribe_records_the_model_that_answered(notebook):
    """A refusal fallback answers as a different model, and the page should say so."""
    with _make_client(notebook, lambda png: ("fallback text", "claude-sonnet-5")) as client:
        sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

        page = client.post(f"/api/pages/{sha}/transcribe").json()

    assert page["model"] == "claude-sonnet-5"


@pytest.mark.unit
def test_retranscribe_clears_the_edited_flag(notebook):
    """Fresh model output is not a human correction, however recently one was made."""
    with _make_client(notebook, lambda png: ("model text", "claude-opus-5")) as client:
        sha = client.get("/api/notebook").json()["pages"][0]["sha256"]
        client.put(f"/api/pages/{sha}", json={"text": "corrected by hand"})

        page = client.post(f"/api/pages/{sha}/transcribe").json()

    assert page["status"] == "transcribed"
    assert "edited_at" not in json.loads(notebook.state_file.read_text())["pages"][sha]


@pytest.mark.unit
def test_retranscribe_rebuilds_the_notebook_markdown(notebook):
    with _make_client(notebook, lambda png: ("rebuilt into the vault", "claude-opus-5")) as client:
        sha = client.get("/api/notebook").json()["pages"][0]["sha256"]
        client.post(f"/api/pages/{sha}/transcribe")

    assert "rebuilt into the vault" in (notebook.vault_dir / "field-notes.md").read_text()


@pytest.mark.unit
def test_retranscribe_resolves_a_recorded_failure(notebook):
    """A refused page is the likeliest thing to be retried, so it must come back clean."""
    with _make_client(notebook, lambda png: ("second attempt worked", "claude-opus-5")) as client:
        sha = client.get("/api/notebook").json()["pages"][1]["sha256"]
        state = State(notebook.state_file)
        state.put_failure(sha, "refused", "TranscriptionRefused: model refused page")
        state.save()

        page = client.post(f"/api/pages/{sha}/transcribe").json()

    assert page["status"] == "transcribed"
    assert page["failure"] is None


@pytest.mark.unit
def test_failed_retranscribe_keeps_the_existing_text(notebook):
    """A page holding a good transcription must not be emptied by a failed retry."""

    def refuse(png):
        raise TranscriptionRefused("model refused page")

    with _make_client(notebook, refuse) as client:
        sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

        response = client.post(f"/api/pages/{sha}/transcribe")

    assert response.status_code == 502
    assert "TranscriptionRefused" in response.json()["detail"]
    assert State(notebook.state_file).get_text(sha) == "page one text"


@pytest.mark.unit
def test_failed_retranscribe_keeps_a_human_correction(notebook):
    """The one irreplaceable thing on a page is the text a person typed."""

    def fail(png):
        raise TranscriptionError("model returned no text")

    with _make_client(notebook, fail) as client:
        sha = client.get("/api/notebook").json()["pages"][0]["sha256"]
        client.put(f"/api/pages/{sha}", json={"text": "typed out by hand"})

        client.post(f"/api/pages/{sha}/transcribe")

        assert client.get("/api/notebook").json()["pages"][0]["status"] == "edited"

    assert State(notebook.state_file).get_text(sha) == "typed out by hand"


@pytest.mark.unit
def test_missing_credentials_are_reported_rather_than_raised(notebook):
    def unauthenticated(png):
        raise CredentialsMissing("no Claude credentials found")

    with _make_client(notebook, unauthenticated) as client:
        sha = client.get("/api/notebook").json()["pages"][0]["sha256"]

        response = client.post(f"/api/pages/{sha}/transcribe")

    assert response.status_code == 503
    assert "credentials" in response.json()["detail"]


@pytest.mark.unit
def test_retranscribing_an_unknown_page_returns_404(client):
    assert client.post("/api/pages/deadbeef/transcribe").status_code == 404


@pytest.mark.unit
def test_opening_the_review_ui_never_builds_an_api_client(notebook, monkeypatch):
    """Reviewing is free until the reviewer asks to spend: no client, no credential check."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        anthropic, "Anthropic", lambda *a, **k: pytest.fail("built an API client at startup")
    )

    with TestClient(create_app(notebook)) as client:
        assert client.get("/api/notebook").status_code == 200


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
