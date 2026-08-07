"""Pydantic-validated configuration loaded from a YAML file."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class Config(BaseModel):
    """Run configuration. Unknown keys are rejected so typos surface immediately."""

    model_config = ConfigDict(extra="forbid")

    input_dir: Path
    vault_dir: Path
    model: str = "claude-opus-5"
    max_tokens: int = 16000
    state_file: Path = Path("state.json")

    @field_validator("input_dir", "vault_dir", "state_file")
    @classmethod
    def _expand_user(cls, value: Path) -> Path:
        return value.expanduser()


def load_config(path: Path) -> Config:
    """Read and validate a config.yaml. Raises pydantic.ValidationError on bad input."""
    data = yaml.safe_load(path.read_text()) or {}
    return Config.model_validate(data)
