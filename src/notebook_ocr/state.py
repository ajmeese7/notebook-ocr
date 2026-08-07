"""Processed-page cache keyed by image content hash.

Storing each page's transcription (not just a "seen" flag) is what makes re-runs both
cheap and complete: a re-run re-transcribes only new images, yet the assembled notebook
always contains every page. Keyed by SHA-256 of the file bytes, so identical content
resolves to the same transcription regardless of folder or filename.
"""

import json
from datetime import datetime
from pathlib import Path

_VERSION = 1


class State:
    """Load, query, and persist the transcription cache in state.json."""

    def __init__(self, path: Path):
        self.path = path
        self._pages: dict[str, dict[str, str]] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            self._pages = raw.get("pages", {})

    def get_text(self, sha256: str) -> str | None:
        """Cached transcription for this content hash, or None if not yet processed."""
        entry = self._pages.get(sha256)
        return entry["text"] if entry else None

    def put(self, sha256: str, text: str, model: str) -> None:
        """Record a fresh transcription with the model and a local-time timestamp."""
        self._pages[sha256] = {
            "text": text,
            "model": model,
            "transcribed_at": datetime.now().astimezone().isoformat(),
        }

    def save(self) -> None:
        """Persist atomically via a temp file so a crash mid-write never corrupts state.json."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"version": _VERSION, "pages": self._pages}
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        tmp.replace(self.path)
