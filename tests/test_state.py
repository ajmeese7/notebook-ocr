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
def test_notebook_title_roundtrips_across_instances(tmp_path):
    path = tmp_path / "state.json"
    state = State(path)
    state.put_title("/photos/cyiq_green", "CYIQ, green cover")
    state.save()

    assert State(path).get_title("/photos/cyiq_green") == "CYIQ, green cover"


@pytest.mark.unit
def test_unset_notebook_title_is_none(tmp_path):
    assert State(tmp_path / "state.json").get_title("/photos/cyiq_green") is None


@pytest.mark.unit
def test_blank_title_clears_a_previous_rename(tmp_path):
    """Clearing the field in the UI must restore the folder name, not store an empty label."""
    state = State(tmp_path / "state.json")
    state.put_title("/photos/cyiq_green", "CYIQ")
    state.put_title("/photos/cyiq_green", "   ")

    assert state.get_title("/photos/cyiq_green") is None


@pytest.mark.unit
def test_titles_are_scoped_to_the_source_folder(tmp_path):
    state = State(tmp_path / "state.json")
    state.put_title("/photos/notes", "Work")

    assert state.get_title("/archive/notes") is None


@pytest.mark.unit
def test_state_file_without_notebooks_key_still_loads(tmp_path):
    """State files written before titles existed must keep working untouched."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1, "pages": {"h1": {"text": "page one"}}}))

    state = State(path)

    assert state.get_text("h1") == "page one"
    assert state.get_title("/photos/notes") is None


@pytest.mark.unit
def test_save_is_atomic_leaving_no_temp_file(tmp_path):
    path = tmp_path / "state.json"
    state = State(path)
    state.put("h1", "text", "claude-opus-5")
    state.save()

    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
