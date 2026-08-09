"""Local review server: page image beside its transcription, with editable text.

Spot-checking a transcription needs the image the *model* saw, not the raw photo — a
wrong line is as often a preprocessing failure (over-crop, bad deskew) as a model one.
So `/api/pages/{sha256}/image` serves the preprocessed PNG that `run` would send.

Corrections are written back into `state.json` keyed by image hash, which is what makes
them durable: the next `run` reads the cache and reuses the correction instead of
overwriting it. The notebook markdown is rebuilt on every save so the vault never lags
behind what the reviewer sees.

The server never calls the Claude API; reviewing cannot cost money. It does bind every
interface by default so a page can be checked from a phone on the same network, and it is
unauthenticated: anyone who can reach the port can read the notebook and rewrite it.
"""

import socket
import sys
import webbrowser
from datetime import date
from functools import lru_cache, partial
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .config import Config
from .discover import DiscoveredImage, discover
from .preprocess import preprocess_image
from .state import State
from .vault import PageEntry, render_notebook

_UI_FILE = Path(__file__).parent / "review.html"

# Preprocessing a page costs a few hundred milliseconds, and reviewers page back and
# forth constantly. Cache by content hash: identical bytes always preprocess identically.
# Bounded because each entry holds a multi-megabyte PNG in memory; this covers the window
# of pages anyone flips through in one sitting without pinning a whole notebook in RAM.
_IMAGE_CACHE_SIZE = 24

# Bind addresses that mean "every interface", so the printed URL and the exposure notice
# are chosen by what the server is actually reachable on.
_ALL_INTERFACES = frozenset({"0.0.0.0", "::", ""})

# TEST-NET-1 (RFC 5737): reserved for documentation and never routed to a real host, so
# the route lookup cannot accidentally contact anything.
_ROUTE_PROBE_ADDRESS = "192.0.2.1"
_ROUTE_PROBE_PORT = 1


class Page(BaseModel):
    """One reviewable page: what to show, and whether it needs attention."""

    page_number: int
    sha256: str
    filename: str
    capture_time: str
    text: str
    # "transcribed" (model output as-is), "edited" (human-corrected), or "missing"
    # (never successfully transcribed, so `run` marked it as failed in the markdown).
    status: str
    model: str | None = None


class Correction(BaseModel):
    """A human-edited transcription submitted from the review UI."""

    text: str = Field(min_length=1)


class Notebook(BaseModel):
    """Everything the UI needs to render a review session."""

    notebook: str
    source_dir: str
    vault_file: str
    pages: list[Page]


@lru_cache(maxsize=_IMAGE_CACHE_SIZE)
def _preprocessed_png(path: Path, sha256: str) -> bytes:
    """Preprocessed PNG for one page. `sha256` is the cache key: same bytes, same output."""
    return preprocess_image(path)


def _status(text: str | None, entry: dict[str, str]) -> str:
    if text is None:
        return "missing"
    return "edited" if entry.get("edited_at") else "transcribed"


def _to_page(page_number: int, image: DiscoveredImage, state: State) -> Page:
    entry = state.get_entry(image.sha256) or {}
    text = state.get_text(image.sha256)
    return Page(
        page_number=page_number,
        sha256=image.sha256,
        filename=image.path.name,
        capture_time=image.capture_time.isoformat(sep=" ", timespec="seconds"),
        text=text or "",
        status=_status(text, entry),
        model=entry.get("model"),
    )


def create_app(config: Config) -> FastAPI:
    """Build the review app over one config's input folder, vault, and state file."""
    source_dir = config.input_dir.resolve()
    app = FastAPI(title=f"notebook-ocr review: {source_dir.name}")

    # Discovery hashes every file in the folder, so do it once at startup rather than per
    # request. Adding photos mid-review means restarting the server, which is the honest
    # tradeoff: page numbers would otherwise shift under an open editor.
    images = discover(source_dir)
    # Duplicate photos share a hash and therefore a transcription; the first occurrence
    # wins for lookup, exactly as the cache already treats them during a run.
    by_hash: dict[str, tuple[int, DiscoveredImage]] = {}
    for number, image in enumerate(images, start=1):
        by_hash.setdefault(image.sha256, (number, image))

    def _lookup(sha256: str) -> tuple[int, DiscoveredImage]:
        found = by_hash.get(sha256)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no page with hash {sha256}")
        return found

    def _rebuild_markdown(state: State) -> Path:
        """Rewrite the notebook .md from current state so the vault matches the edits."""
        pages = [
            PageEntry(
                page_number=number,
                filename=image.path.name,
                capture_time=image.capture_time,
                text=state.get_text(image.sha256) or _missing_marker(image.path.name),
            )
            for number, image in enumerate(images, start=1)
        ]
        return render_notebook(config.vault_dir, source_dir, pages, date.today())

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_UI_FILE, media_type="text/html")

    @app.get("/api/notebook")
    def read_notebook() -> Notebook:
        state = State(config.state_file)
        return Notebook(
            notebook=source_dir.name,
            source_dir=str(source_dir),
            vault_file=str(config.vault_dir / f"{source_dir.name}.md"),
            pages=[_to_page(number, image, state) for number, image in enumerate(images, start=1)],
        )

    @app.get("/api/pages/{sha256}/image")
    def read_page_image(sha256: str) -> Response:
        _, image = _lookup(sha256)
        return Response(
            content=_preprocessed_png(image.path, image.sha256),
            media_type="image/png",
            headers={"Cache-Control": "no-cache"},
        )

    @app.put("/api/pages/{sha256}")
    def save_correction(sha256: str, correction: Correction) -> Page:
        page_number, image = _lookup(sha256)
        # Re-read rather than holding one State: the file is the source of truth, and a
        # `run` may have written to it since this server started.
        state = State(config.state_file)
        state.put_correction(sha256, correction.text)
        state.save()
        _rebuild_markdown(state)
        return _to_page(page_number, image, state)

    return app


def _missing_marker(filename: str) -> str:
    return f"<!-- NOT TRANSCRIBED: {filename} -->"


def lan_address() -> str | None:
    """Best-effort IPv4 this machine is reachable at, for opening the UI on another device.

    Connecting a UDP socket sends no packets; it only asks the kernel which local address
    would be used to reach the outside, which is the one a phone on the same network can
    dial. None when there is no route (offline, or loopback-only).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect((_ROUTE_PROBE_ADDRESS, _ROUTE_PROBE_PORT))
            return probe.getsockname()[0]
        except OSError:
            return None


def browsable_url(host: str, port: int) -> str:
    """A URL this machine's browser can open.

    A wildcard bind address is not somewhere a browser can connect, so it maps to loopback
    rather than being pasted into a URL as-is.
    """
    return f"http://{'127.0.0.1' if host in _ALL_INTERFACES else host}:{port}"


def serve(config: Config, host: str, port: int, open_browser: bool) -> None:
    """Run the review UI until interrupted."""
    if not config.input_dir.is_dir():
        raise SystemExit(f"input_dir is not a directory: {config.input_dir}")

    app = create_app(config)
    on_every_interface = host in _ALL_INTERFACES
    local_url = browsable_url(host, port)

    # stderr, like the rest of the CLI's progress output: stdout is block-buffered when
    # redirected, which would hide the URL and the warning until the server exits.
    say = partial(print, file=sys.stderr)
    say(f"Review UI for {config.input_dir.resolve().name} at {local_url} (ctrl-c to stop)")
    if on_every_interface:
        address = lan_address()
        if address:
            say(f"  from another device on this network: http://{address}:{port}")
        # Anyone who can reach this port can read every page and rewrite the notes, so do
        # not let that be a surprise.
        say("  no password: anyone on this network can read and edit these notes")
        say("  restrict it with: --host 127.0.0.1")

    if open_browser:
        webbrowser.open(local_url)
    uvicorn.run(app, host=host, port=port, log_level="warning")
