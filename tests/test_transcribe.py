import pytest

from notebook_ocr.transcribe import (
    TranscriptionError,
    TranscriptionRefused,
    TranscriptionTruncated,
    interpret_response,
)


@pytest.mark.unit
def test_returns_stripped_text_on_normal_completion():
    text = interpret_response("end_turn", "  # Page one\n\nnotes  ", 16000, None)

    assert text == "# Page one\n\nnotes"


@pytest.mark.unit
def test_raises_on_refusal():
    with pytest.raises(TranscriptionRefused):
        interpret_response("refusal", "", 16000, {"category": "cyber"})


@pytest.mark.unit
def test_truncated_output_raises_instead_of_returning_partial_text():
    """Regression: a partial page was previously cached in state.json as if complete."""
    with pytest.raises(TranscriptionTruncated):
        interpret_response("max_tokens", "The first half of the pa", 16000, None)


@pytest.mark.unit
def test_truncation_message_names_the_limit_and_the_fix():
    with pytest.raises(TranscriptionTruncated, match="16000"):
        interpret_response("max_tokens", "partial", 16000, None)


@pytest.mark.unit
def test_empty_response_raises_rather_than_caching_blank_page():
    with pytest.raises(TranscriptionError):
        interpret_response("end_turn", "   ", 16000, None)


@pytest.mark.unit
def test_failure_types_share_a_common_base_so_callers_catch_one_thing():
    assert issubclass(TranscriptionRefused, TranscriptionError)
    assert issubclass(TranscriptionTruncated, TranscriptionError)
