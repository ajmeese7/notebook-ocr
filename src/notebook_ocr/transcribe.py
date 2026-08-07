"""Claude vision call: one cleaned image in, verbatim markdown out."""

import base64

import anthropic

TRANSCRIPTION_PROMPT = (
    "Transcribe this handwritten notebook page to markdown, verbatim.\n"
    "Preserve headings, lists, and structure. Do not summarize, correct, or add commentary.\n"
    "Mark anything you genuinely cannot read as [?]. Output only the transcription."
)


class TranscriptionError(RuntimeError):
    """A page could not be transcribed into a result worth caching."""


class TranscriptionRefused(TranscriptionError):
    """The model declined to transcribe a page (stop_reason == 'refusal')."""


class TranscriptionTruncated(TranscriptionError):
    """Output hit max_tokens, so the transcription is cut off mid-page."""


class CredentialsMissing(RuntimeError):
    """No API credentials are resolvable, so every page would fail identically."""


def interpret_response(
    stop_reason: str | None, text: str, max_tokens: int, details: object
) -> str:
    """Validate a finished response and return the transcription, or raise.

    Split out from the API call so the failure rules are exercised directly, without
    standing up a fake client.
    """
    # Check stop_reason before trusting content: on a refusal, content is empty.
    if stop_reason == "refusal":
        raise TranscriptionRefused(f"model refused page: {details}")

    # Thinking shares the max_tokens budget, so a dense page can run out mid-sentence.
    # Never cache a partial transcription — it would silently persist as if complete.
    if stop_reason == "max_tokens":
        raise TranscriptionTruncated(
            f"transcription hit max_tokens ({max_tokens}) and is incomplete; "
            "raise max_tokens in config.yaml and re-run"
        )

    stripped = text.strip()
    if not stripped:
        raise TranscriptionError(f"model returned no text (stop_reason={stop_reason})")
    return stripped


class Transcriber:
    """Wraps the Anthropic client. Credentials resolve from the environment (bare client)."""

    def __init__(self, model: str, max_tokens: int):
        # Constructing the client never fails; credential resolution is deferred to the
        # first request, so the missing-credentials check lives in transcribe().
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def transcribe(self, png_bytes: bytes) -> str:
        """Send one PNG page and return its markdown transcription."""
        # Image block before the text block; base64 must have no embedded newlines.
        encoded = base64.standard_b64encode(png_bytes).decode("utf-8")
        try:
            response = self._request(encoded)
        except TypeError as error:
            # The SDK raises TypeError at request time when it cannot resolve any
            # credential source (env var, auth token, or `ant auth login` profile).
            raise CredentialsMissing(
                "no Claude credentials found. Set ANTHROPIC_API_KEY in your environment "
                "or in a .env file (see .env.example), or run `ant auth login`."
            ) from error

        text = "".join(block.text for block in response.content if block.type == "text")
        return interpret_response(
            response.stop_reason, text, self.max_tokens, getattr(response, "stop_details", None)
        )

    def _request(self, encoded: str) -> anthropic.types.Message:
        """Issue the vision request and return the completed message."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": encoded,
                            },
                        },
                        {"type": "text", "text": TRANSCRIPTION_PROMPT},
                    ],
                }
            ],
        ) as stream:
            return stream.get_final_message()
