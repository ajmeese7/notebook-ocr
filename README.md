# notebook-ocr

Point it at a folder of notebook page photos; it transcribes each page with the Claude API and writes one markdown file per notebook into a vault. Handwriting is messy half-cursive, so a vision LLM does the transcription (classical OCR fails on cursive).

## How it works

- **One folder = one notebook.** The output file is named after the folder: `<folder>.md`.
- **No renaming required.** Page order is derived from capture time (EXIF `DateTimeOriginal`, falling back to file mtime), with filename as a stable tiebreak. Just photograph pages in order.
- **Local cleanup first.** Each image is deskewed, gently auto-cropped, and contrast-boosted (OpenCV) before it is sent as PNG to Claude.
- **Idempotent re-runs.** Every page's transcription is cached in `state.json`, keyed by the SHA-256 of the image bytes. Re-running only transcribes new images, but always rebuilds the complete notebook.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13 (uv will fetch one).

```sh
uv sync
```

Provide Claude credentials via the environment — `ANTHROPIC_API_KEY`, or an [`ant auth login`](https://platform.claude.com) profile. Nothing secret goes in config. A `.env` file in the working directory is loaded automatically (it is gitignored); an already-exported variable takes precedence.

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

Photos, transcriptions (`vault/`), and `state.json` are gitignored. If you point `input_dir` or `vault_dir` somewhere else, make sure it stays covered by `.gitignore`. The API key is read from the environment and never written to disk.

## Development

```sh
uv run pytest
```
