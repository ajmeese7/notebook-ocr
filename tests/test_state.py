from notebook_ocr.state import State


def test_put_then_get_roundtrips_text(tmp_path):
    state = State(tmp_path / "state.json")
    state.put("abc", "hello page", "claude-opus-5")

    assert state.get_text("abc") == "hello page"


def test_get_unknown_hash_returns_none(tmp_path):
    state = State(tmp_path / "state.json")

    assert state.get_text("missing") is None


def test_save_persists_across_instances(tmp_path):
    path = tmp_path / "state.json"
    first = State(path)
    first.put("h1", "page one", "claude-opus-5")
    first.save()

    reloaded = State(path)

    assert reloaded.get_text("h1") == "page one"


def test_save_records_model_and_timestamp(tmp_path):
    import json

    path = tmp_path / "state.json"
    state = State(path)
    state.put("h1", "text", "claude-opus-5")
    state.save()

    entry = json.loads(path.read_text())["pages"]["h1"]
    assert entry["model"] == "claude-opus-5"
    assert entry["transcribed_at"]
