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
    # Optional second model tried only when the primary refuses a page. The usage policy
    # is shared across models, so a fallback is not guaranteed to clear a refusal, but a
    # smaller model without the Opus-tier dual-use measures often will. Left unset by
    # default so nothing changes for callers who do not want it.
    fallback_model: str | None = None
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
