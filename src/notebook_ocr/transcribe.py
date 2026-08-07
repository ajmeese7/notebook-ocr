"""Claude vision call: one cleaned image in, verbatim markdown out."""

import base64

import anthropic

TRANSCRIPTION_PROMPT = (
    "Transcribe this handwritten notebook page to markdown, verbatim.\n"
    "Preserve headings, lists, and structure. Do not summarize, correct, or add commentary.\n"
    "Mark anything you genuinely cannot read as [?]. Output only the transcription."
)


class TranscriptionRefused(RuntimeError):
    """Raised when the model declines to transcribe a page (stop_reason == 'refusal')."""


class Transcriber:
    """Wraps the Anthropic client. Credentials resolve from the environment (bare client)."""

    def __init__(self, model: str, max_tokens: int):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def transcribe(self, png_bytes: bytes) -> str:
        """Send one PNG page and return its markdown transcription."""
        # Image block before the text block; base64 must have no embedded newlines.
        encoded = base64.standard_b64encode(png_bytes).decode("utf-8")
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
            response = stream.get_final_message()

        # Check stop_reason before reading content: on a refusal, content is empty.
        if response.stop_reason == "refusal":
            raise TranscriptionRefused(f"model refused page: {response.stop_details}")

        return "".join(block.text for block in response.content if block.type == "text").strip()
