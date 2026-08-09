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

    def get_entry(self, sha256: str) -> dict[str, str] | None:
        """The full stored record for this content hash, or None if absent or malformed."""
        entry = self._pages.get(sha256)
        return entry if isinstance(entry, dict) else None

    def get_text(self, sha256: str) -> str | None:
        """Cached transcription for this content hash, or None if not usable.

        A malformed entry (hand-edited file, interrupted older write) is treated as a
        miss rather than an error, so the page is simply transcribed again.
        """
        entry = self.get_entry(sha256)
        if entry is None:
            return None
        text = entry.get("text")
        return text if isinstance(text, str) and text.strip() else None

    def put(self, sha256: str, text: str, model: str) -> None:
        """Record a fresh transcription with the model and a local-time timestamp."""
        self._pages[sha256] = {
            "text": text,
            "model": model,
            "transcribed_at": datetime.now().astimezone().isoformat(),
        }

    def put_correction(self, sha256: str, text: str) -> dict[str, str]:
        """Overwrite a page's text with a human correction, returning the updated entry.

        The original `model` and `transcribed_at` are preserved and an `edited_at` stamp
        is added, so a corrected page stays distinguishable from raw model output. Because
        the correction lives in the cache rather than only in the markdown, a later `run`
        reuses it instead of overwriting it with the model's version.
        """
        entry = dict(self.get_entry(sha256) or {})
        entry["text"] = text
        entry["edited_at"] = datetime.now().astimezone().isoformat()
        self._pages[sha256] = entry
        return entry

    def save(self) -> None:
        """Persist atomically via a temp file so a crash mid-write never corrupts state.json."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"version": _VERSION, "pages": self._pages}
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        tmp.replace(self.path)
