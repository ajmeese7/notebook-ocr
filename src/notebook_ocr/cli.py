"""`notebook-ocr run -c config.yaml` — the single command over one folder."""

import argparse
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from .config import load_config
from .discover import discover
from .preprocess import preprocess_image
from .state import State
from .transcribe import Transcriber
from .vault import PageEntry, build_markdown, write_notebook


def run(config_path: Path) -> Path | None:
    """Execute the pipeline for one input folder; returns the written .md path (None if empty)."""
    config = load_config(config_path)

    if not config.input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {config.input_dir}")

    images = discover(config.input_dir)
    if not images:
        print(f"No images found in {config.input_dir}", file=sys.stderr)
        return None

    state = State(config.state_file)
    transcriber: Transcriber | None = None  # created lazily; a fully-cached run needs no API client
    pages: list[PageEntry] = []

    for page_number, image in enumerate(images, start=1):
        text = state.get_text(image.sha256)
        if text is None:
            if transcriber is None:
                transcriber = Transcriber(config.model, config.max_tokens)
            print(f"[{page_number}/{len(images)}] transcribing {image.path.name}", file=sys.stderr)
            png = preprocess_image(image.path)
            text = transcriber.transcribe(png)
            state.put(image.sha256, text, config.model)
            state.save()  # persist incrementally so a crash mid-run never re-bills done pages
        else:
            print(f"[{page_number}/{len(images)}] cached    {image.path.name}", file=sys.stderr)

        pages.append(
            PageEntry(
                page_number=page_number,
                filename=image.path.name,
                capture_time=image.capture_time,
                text=text,
            )
        )

    source_dir = config.input_dir.resolve()
    notebook = source_dir.name
    content = build_markdown(notebook, source_dir, pages, date.today())
    out_path = write_notebook(config.vault_dir, notebook, content)
    print(f"Wrote {len(pages)} page(s) to {out_path}", file=sys.stderr)
    return out_path


def main(argv: list[str] | None = None) -> None:
    # Load a local .env if present so ANTHROPIC_API_KEY resolves without exporting it.
    # An already-set environment variable wins (load_dotenv does not override by default).
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="notebook-ocr",
        description="Transcribe a folder of notebook page photos into one markdown file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="transcribe one folder into a notebook .md")
    run_parser.add_argument(
        "-c", "--config", type=Path, default=Path("config.yaml"), help="path to config.yaml"
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.config)
