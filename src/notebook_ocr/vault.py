"""Assemble transcribed pages into one markdown file per notebook."""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class PageEntry:
    """A single transcribed page, ready to render into the notebook markdown."""

    page_number: int
    filename: str
    capture_time: datetime
    text: str


def build_markdown(
    notebook: str, source_dir: Path, pages: list[PageEntry], generated: date
) -> str:
    """Render YAML front-matter plus per-page HTML-comment markers keeping every line traceable."""
    lines = [
        "---",
        f"notebook: {notebook}",
        f"pages: {len(pages)}",
        f"source_dir: {source_dir}",
        f"generated: {generated.isoformat()}",
        "---",
        "",
    ]
    for page in pages:
        timestamp = page.capture_time.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"<!-- page {page.page_number:03d} ({page.filename}, {timestamp}) -->")
        lines.append("")
        lines.append(page.text.rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_notebook(vault_dir: Path, notebook: str, content: str) -> Path:
    """Write <notebook>.md into the vault, creating the directory if needed."""
    vault_dir.mkdir(parents=True, exist_ok=True)
    out_path = vault_dir / f"{notebook}.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def render_notebook(
    vault_dir: Path, source_dir: Path, pages: list[PageEntry], generated: date
) -> Path:
    """Build and write one notebook, naming it after the source folder.

    Shared by the transcription run and the review server so a saved correction produces
    a byte-identical file to a fresh run.
    """
    notebook = source_dir.name
    content = build_markdown(notebook, source_dir, pages, generated)
    return write_notebook(vault_dir, notebook, content)
