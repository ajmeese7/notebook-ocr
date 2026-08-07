import json

import pytest

from notebook_ocr.state import State


@pytest.mark.unit
def test_put_then_get_roundtrips_text(tmp_path):
    state = State(tmp_path / "state.json")
    state.put("abc", "hello page", "claude-opus-5")

    assert state.get_text("abc") == "hello page"


@pytest.mark.unit
def test_get_unknown_hash_returns_none(tmp_path):
    state = State(tmp_path / "state.json")

    assert state.get_text("missing") is None


@pytest.mark.unit
def test_save_persists_across_instances(tmp_path):
    path = tmp_path / "state.json"
    first = State(path)
    first.put("h1", "page one", "claude-opus-5")
    first.save()

    reloaded = State(path)

    assert reloaded.get_text("h1") == "page one"


@pytest.mark.unit
def test_save_records_model_and_timestamp(tmp_path):
    path = tmp_path / "state.json"
    state = State(path)
    state.put("h1", "text", "claude-opus-5")
    state.save()

    entry = json.loads(path.read_text())["pages"]["h1"]
    assert entry["model"] == "claude-opus-5"
    assert entry["transcribed_at"]


@pytest.mark.unit
def test_malformed_entry_is_treated_as_a_cache_miss(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1, "pages": {"h1": {"model": "x"}}}))

    assert State(path).get_text("h1") is None


@pytest.mark.unit
def test_blank_cached_text_is_treated_as_a_cache_miss(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1, "pages": {"h1": {"text": "   "}}}))

    assert State(path).get_text("h1") is None


@pytest.mark.unit
def test_save_is_atomic_leaving_no_temp_file(tmp_path):
    path = tmp_path / "state.json"
    state = State(path)
    state.put("h1", "text", "claude-opus-5")
    state.save()

    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
