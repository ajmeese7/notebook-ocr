# notebook-ocr

Point it at a folder of notebook page photos; it transcribes each page with the Claude API and writes one markdown file per notebook into a vault. Handwriting is messy half-cursive, so a vision LLM does the transcription (classical OCR fails on cursive).

## How it works

- **One folder = one notebook.** The output file is named after the folder: `<folder>.md`.
- **No renaming required.** Page order is derived from capture time (EXIF `DateTimeOriginal`, falling back to file mtime), with filename as a stable tiebreak. Just photograph pages in order.
- **Local cleanup first.** Each image is deskewed, gently auto-cropped, contrast-boosted, and downscaled to a 2576px long edge (OpenCV) before it is sent as PNG to Claude. The downscale matches the model's vision resolution and keeps the request under the API's 5 MB per-image limit, which a straight-from-the-phone 12 MP page would otherwise exceed.
- **Idempotent re-runs.** Every page's transcription is cached in `state.json`, keyed by the SHA-256 of the image bytes. Re-running only transcribes new images, but always rebuilds the complete notebook.
- **Failures are per-page.** A page that is refused, truncated, or empty is not cached and does not abort the run: it is marked inline in the markdown, listed on stderr, and the command exits non-zero. Re-running retries only those pages. The reason is recorded in `state.json` (with no text, so the page stays a cache miss) so `review` can show which pages failed and why.
- **Spot-checkable.** `notebook-ocr review` opens a local side-by-side view of each page image and its transcription, so a misread line is obvious. Corrections save back into the cache and survive later runs.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13 (uv will fetch one).

```sh
uv sync
```

The Claude API key comes from the `ANTHROPIC_API_KEY` environment variable. Either export it yourself, or put it in a `.env` file in the working directory — `cp .env.example .env` — which the CLI loads at startup. An already-exported variable takes precedence over `.env`.

## Usage

```sh
cp config.example.yaml config.yaml   # edit paths; config.yaml is gitignored
uv run notebook-ocr run -c config.yaml
```

## Reviewing

Transcription errors are only visible against the page they came from, so `review` serves a local side-by-side view and opens it in your browser:

```sh
uv run notebook-ocr review -c config.yaml     # --port 8420, --host, --no-browser
```

The left pane shows the **preprocessed** image, the exact bytes sent to Claude, so a bad line is attributable to either the model or the deskew/crop that fed it. The right pane is the transcription, editable in place. `j`/`k` (or the arrow keys) change page, ctrl-s saves, and clicking the image toggles between fit-to-pane and full resolution. Page markers show status at a glance: grey for model output, green for human-edited, yellow for not transcribed yet, and red for a page a run tried and could not transcribe. The filename in the toolbar takes the same colour, and names the reason (`refused`, `truncated`, `api_error`) beside it: a refusal will not clear by re-running, while a truncation usually will.

Drawings are transcribed as inline `<svg>`, which is unreadable as source, so **Drawings** (`p`) splits the transcription pane and renders each one below the text, on a light card (the strokes are black on transparent, invisible against the dark editor). The divider drags, and double-clicking it restores the default height. Clicking a rendered drawing selects its markup in the transcription; an SVG the browser cannot parse says so on the card instead of showing a blank. Each drawing is rendered through a `data:` URL in an `<img>`, so model output cannot run script in the review page.

The sidebar drags to any width (double-click the divider to reset) and `[` collapses it. Its heading is editable: type a friendlier name for a folder like `cyiq_green` and it is saved in `state.json`. That title is **display only** — the vault file stays `<folder>.md` and page order is unchanged, so renaming can never orphan a notebook. Clearing the field restores the folder name. Width, collapsed state, and whether the drawings pane is open persist per browser.

Saving writes the correction into `state.json` keyed by image hash and immediately rebuilds the notebook `.md`. Because the correction lives in the cache, a later `run` reuses it instead of overwriting it with the model's version.

**Re-transcribe** sends the current page to the model again and replaces its text. The cache is keyed by image bytes, so nothing about a page ever invalidates it on its own: a prompt change, a different model, or a crop you have since fixed are all reasons to redo a page that is technically already done. This is the only action in the UI that calls the Claude API and the only one that costs money, so it is behind a confirmation naming the file, the model, and what will be overwritten (a saved correction is called out separately, since that is the one thing here a re-run cannot reproduce). A failed attempt changes nothing: the existing text or correction stays, and the error is shown in the toolbar.

Everything else in the UI is local and free. The server binds every interface by default, so you can check pages from a phone or tablet on the same network. On startup it prints the address to use:

```
Review UI for field-notes-2024 at http://127.0.0.1:8420 (ctrl-c to stop)
  from another device on this network: http://192.168.1.204:8420
  no password: anyone on this network can read, edit, and re-transcribe these
  restrict it with: --host 127.0.0.1
```

There is no authentication, so anyone who can reach the port can read the notebook, rewrite it, and spend your API credit re-transcribing it. That is fine on a home network and a bad idea on a shared or public one, where `--host 127.0.0.1` keeps it on this machine.

## Config

```yaml
input_dir: ./photos           # one folder = one notebook; output is <folder-name>.md
vault_dir: ./vault            # where the .md file is written
model: claude-opus-5
max_tokens: 16000
state_file: ./state.json
```

## Privacy

Photos, transcriptions (`vault/`), and `state.json` are gitignored — note that `state.json` caches full page text, so it is as sensitive as the notebook itself. If you point `input_dir`, `vault_dir`, or `state_file` somewhere else, make sure it stays covered by `.gitignore`. If you use a `.env`, your key is stored there in plaintext; it is gitignored, but keep it out of backups and shared drives.

## Development

```sh
uv run pytest
```
