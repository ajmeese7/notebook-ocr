import pytest
from pydantic import ValidationError

from notebook_ocr.config import load_config


@pytest.mark.unit
def test_loads_valid_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "input_dir: ./photos\n"
        "vault_dir: ./vault\n"
        "model: claude-opus-5\n"
        "max_tokens: 16000\n"
        "state_file: ./state.json\n"
    )

    config = load_config(cfg)

    assert config.model == "claude-opus-5"
    assert config.max_tokens == 16000


@pytest.mark.unit
def test_applies_defaults_when_optional_fields_omitted(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("input_dir: ./photos\nvault_dir: ./vault\n")

    config = load_config(cfg)

    assert config.model == "claude-opus-5"
    assert config.max_tokens == 16000
    assert config.state_file.name == "state.json"


@pytest.mark.unit
def test_rejects_unknown_key(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("input_dir: ./photos\nvault_dir: ./vault\napi_key: sk-should-not-be-here\n")

    with pytest.raises(ValidationError):
        load_config(cfg)


@pytest.mark.unit
def test_expands_home_directory(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("input_dir: ~/photos\nvault_dir: ~/vault\n")

    config = load_config(cfg)

    assert "~" not in str(config.input_dir)
