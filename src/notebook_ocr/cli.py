"""`notebook-ocr run -c config.yaml` — the single command over one folder."""

import argparse
import sys
from datetime import date
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from .config import load_config
from .discover import discover
from .preprocess import preprocess_image
from .review import serve
from .state import State
from .transcribe import CredentialsMissing, TranscriberPool, TranscriptionError
from .vault import PageEntry, render_notebook

# A bad key, model name, or permission fails identically on every page, so abort the run
# instead of burning through the whole folder to collect the same error N times.
_FATAL_API_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
)

# Bound to all interfaces so pages can be checked from a phone or tablet on the same
# network, which is where a photo of a page is easiest to compare against. The server is
# unauthenticated and can read every page and edit the notes, so `serve` says plainly what
# is exposed and how to restrict it.
_DEFAULT_REVIEW_HOST = "0.0.0.0"
_DEFAULT_REVIEW_PORT = 8420


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
    # Both created lazily; a fully-cached run needs no API client. The fallback is only
    # built the first time the primary refuses a page.
    transcribers = TranscriberPool(config)
    pages: list[PageEntry] = []
    failures: list[str] = []

    for page_number, image in enumerate(images, start=1):
        text = state.get_text(image.sha256)
        if text is None:
            print(f"[{page_number}/{len(images)}] transcribing {image.path.name}", file=sys.stderr)
            png = preprocess_image(image.path)
            try:
                text, model_used = transcribers.transcribe(png)
            except CredentialsMissing as error:
                raise SystemExit(str(error)) from error
            except _FATAL_API_ERRORS as error:
                raise SystemExit(f"aborting: {type(error).__name__}: {error}") from error
            except (TranscriptionError, anthropic.APIError) as error:
                # Don't cache a result and don't abandon the pages that did work: mark
                # this one inline so a re-run retries only it. The reason is recorded in
                # state (without any text, so the page stays a cache miss) because it is
                # otherwise lost when the run ends, and review has no way to tell a page
                # the model refused from one nobody has transcribed yet.
                print(f"  !! {image.path.name}: {error}", file=sys.stderr)
                failures.append(image.path.name)
                detail = f"{type(error).__name__}: {error}"
                state.put_failure(image.sha256, getattr(error, "reason", "api_error"), detail)
                state.save()
                text = f"<!-- TRANSCRIPTION FAILED: {detail} -->"
            else:
                state.put(image.sha256, text, model_used)
                state.save()  # persist incrementally so a crash never re-bills done pages
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

    out_path = render_notebook(config.vault_dir, config.input_dir.resolve(), pages, date.today())
    print(f"Wrote {len(pages)} page(s) to {out_path}", file=sys.stderr)

    if failures:
        raise SystemExit(
            f"{len(failures)} page(s) failed and are marked in {out_path}: "
            f"{', '.join(failures)}. Re-run to retry only those pages."
        )
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

    review_parser = subparsers.add_parser(
        "review", help="spot-check pages against their images in a local browser UI"
    )
    review_parser.add_argument(
        "-c", "--config", type=Path, default=Path("config.yaml"), help="path to config.yaml"
    )
    review_parser.add_argument(
        "--host",
        default=_DEFAULT_REVIEW_HOST,
        help="bind address; 127.0.0.1 restricts the UI to this machine",
    )
    review_parser.add_argument("--port", type=int, default=_DEFAULT_REVIEW_PORT, help="bind port")
    review_parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.config)
    elif args.command == "review":
        serve(load_config(args.config), args.host, args.port, not args.no_browser)
