# notebook-ocr

Point it at a folder of notebook page photos; it transcribes each page with the Claude API and writes one markdown file per notebook into a vault. Handwriting is messy half-cursive, so a vision LLM does the transcription (classical OCR fails on cursive).

## How it works

- **One folder = one notebook.** The output file is named after the folder: `<folder>.md`.
- **No renaming required.** Page order is derived from capture time (EXIF `DateTimeOriginal`, falling back to file mtime), with filename as a stable tiebreak. Just photograph pages in order.
- **Local cleanup first.** Each image is deskewed, gently auto-cropped, contrast-boosted, and downscaled to a 2576px long edge (OpenCV) before it is sent as PNG to Claude. The downscale matches the model's vision resolution and keeps the request under the API's 5 MB per-image limit, which a straight-from-the-phone 12 MP page would otherwise exceed.
- **Idempotent re-runs.** Every page's transcription is cached in `state.json`, keyed by the SHA-256 of the image bytes. Re-running only transcribes new images, but always rebuilds the complete notebook.
- **Failures are per-page.** A page that is refused, truncated, or empty is not cached and does not abort the run: it is marked inline in the markdown, listed on stderr, and the command exits non-zero. Re-running retries only those pages.

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
