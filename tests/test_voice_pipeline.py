# Checks that STT and TTS both work end-to-end and that the Deepgram fallback
# triggers correctly when Sarvam is unavailable.
#
# All HTTP calls are mocked — these tests validate provider switching logic
# and response parsing, not live API connectivity.

import base64
import logging
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)

from voice.stt import STTProvider
from voice.tts import TTSProvider

DUMMY_AUDIO = b"RIFF....fakeaudiobytes"


# ── STT tests ──────────────────────────────────────────────────────────────────

def test_stt_sarvam_success():
    """When Sarvam returns a transcript, STTProvider must return it directly."""
    with patch("voice.stt.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {
            "transcript": "what is the weather in Tokyo"
        }

        stt = STTProvider()
        result = stt.transcribe(DUMMY_AUDIO)

    assert "Tokyo" in result, f"Expected 'Tokyo' in transcript, got: {result!r}"
    log.info("PASS | test_stt_sarvam_success")


def test_stt_sarvam_fail_deepgram_fallback():
    """When all Sarvam retries fail, STTProvider must transparently switch to Deepgram."""
    deepgram_mock = MagicMock()
    deepgram_mock.status_code = 200
    deepgram_mock.raise_for_status = MagicMock()
    deepgram_mock.json.return_value = {
        "results": {
            "channels": [{"alternatives": [{"transcript": "deepgram result"}]}]
        }
    }

    with patch("voice.stt.requests.post") as mock_post, \
         patch("voice.base.time.sleep"):   # skip backoff waits in tests
        # VOICE_MAX_RETRIES=2 → 3 total Sarvam attempts before fallback fires
        mock_post.side_effect = [
            Exception("Sarvam down"),   # attempt 1
            Exception("Sarvam down"),   # attempt 2 (retry 1)
            Exception("Sarvam down"),   # attempt 3 (retry 2)
            deepgram_mock,              # Deepgram fallback
        ]

        stt = STTProvider()
        result = stt.transcribe(DUMMY_AUDIO)

    assert result == "deepgram result", f"Expected Deepgram transcript, got: {result!r}"
    log.info("PASS | test_stt_sarvam_fail_deepgram_fallback")


def test_stt_both_fail_returns_empty_string():
    """When both providers fail, STTProvider must return '' rather than raising."""
    with patch("voice.stt.requests.post") as mock_post:
        mock_post.side_effect = Exception("All STT providers down")

        stt = STTProvider()
        result = stt.transcribe(DUMMY_AUDIO)

    assert result == "", f"Expected empty string on total failure, got: {result!r}"
    log.info("PASS | test_stt_both_fail_returns_empty_string")


# ── TTS tests ──────────────────────────────────────────────────────────────────

def test_tts_sarvam_success():
    """When Sarvam returns base64 audio, TTSProvider must decode and return bytes."""
    fake_audio_bytes = b"fakeaudiodata"
    fake_b64 = base64.b64encode(fake_audio_bytes).decode()

    with patch("voice.tts.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {"audios": [fake_b64]}

        tts = TTSProvider()
        result = tts.synthesize("Hello Tokyo")

    assert isinstance(result, bytes), "TTS must return bytes"
    assert result == fake_audio_bytes, "Decoded audio must match original bytes"
    log.info("PASS | test_tts_sarvam_success")


def test_tts_sarvam_fail_deepgram_fallback():
    """When all Sarvam retries fail, TTSProvider must switch to Deepgram and return its bytes."""
    deepgram_mock = MagicMock()
    deepgram_mock.status_code = 200
    deepgram_mock.raise_for_status = MagicMock()
    deepgram_mock.content = b"deepgram_audio_bytes"

    with patch("voice.tts.requests.post") as mock_post, \
         patch("voice.base.time.sleep"):   # skip backoff waits in tests
        # VOICE_MAX_RETRIES=2 → 3 total Sarvam attempts before fallback fires
        mock_post.side_effect = [
            Exception("Sarvam TTS down"),  # attempt 1
            Exception("Sarvam TTS down"),  # attempt 2 (retry 1)
            Exception("Sarvam TTS down"),  # attempt 3 (retry 2)
            deepgram_mock,                 # Deepgram fallback
        ]

        tts = TTSProvider()
        result = tts.synthesize("Hello world")

    assert result == b"deepgram_audio_bytes", f"Expected Deepgram audio, got: {result!r}"
    log.info("PASS | test_tts_sarvam_fail_deepgram_fallback")


def test_tts_both_fail_returns_empty_bytes():
    """When both TTS providers fail, TTSProvider must return b'' rather than raising."""
    with patch("voice.tts.requests.post") as mock_post:
        mock_post.side_effect = Exception("All TTS providers down")

        tts = TTSProvider()
        result = tts.synthesize("Some text")

    assert result == b"", f"Expected empty bytes on total failure, got: {result!r}"
    log.info("PASS | test_tts_both_fail_returns_empty_bytes")


# ── Retry behaviour tests ──────────────────────────────────────────────────────
# These verify that retry_call() is wired in — Sarvam fails N times then
# succeeds, proving the backoff loop is active, not just a single-shot try.

def test_stt_sarvam_retries_before_succeeding():
    """STTProvider must retry Sarvam on transient failures and return on eventual success."""
    ok_mock = MagicMock()
    ok_mock.raise_for_status = MagicMock()
    ok_mock.json.return_value = {"transcript": "retry succeeded"}

    with patch("voice.stt.requests.post") as mock_post, \
         patch("voice.base.time.sleep"):          # skip real waits in tests
        # Fail twice, succeed on 3rd attempt
        mock_post.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            ok_mock,
        ]
        stt = STTProvider()
        result = stt.transcribe(DUMMY_AUDIO)

    assert result == "retry succeeded"
    assert mock_post.call_count == 3   # confirms retries happened
    log.info("PASS | test_stt_sarvam_retries_before_succeeding")


def test_tts_sarvam_retries_before_succeeding():
    """TTSProvider must retry Sarvam on transient failures and return on eventual success."""
    import base64 as _b64
    fake_audio = b"retried_audio"
    ok_mock = MagicMock()
    ok_mock.raise_for_status = MagicMock()
    ok_mock.json.return_value = {"audios": [_b64.b64encode(fake_audio).decode()]}

    with patch("voice.tts.requests.post") as mock_post, \
         patch("voice.base.time.sleep"):          # skip real waits in tests
        # Fail twice, succeed on 3rd attempt
        mock_post.side_effect = [
            Exception("timeout"),
            Exception("timeout"),
            ok_mock,
        ]
        tts = TTSProvider()
        result = tts.synthesize("test retry")

    assert result == fake_audio
    assert mock_post.call_count == 3   # confirms retries happened
    log.info("PASS | test_tts_sarvam_retries_before_succeeding")


def test_stt_falls_back_after_all_retries_exhausted():
    """After all Sarvam retries fail, STTProvider must switch to Deepgram exactly once."""
    deepgram_mock = MagicMock()
    deepgram_mock.raise_for_status = MagicMock()
    deepgram_mock.json.return_value = {
        "results": {"channels": [{"alternatives": [{"transcript": "deepgram after retries"}]}]}
    }

    with patch("voice.stt.requests.post") as mock_post, \
         patch("voice.base.time.sleep"):
        # All Sarvam attempts fail (VOICE_MAX_RETRIES=2 → 3 attempts), then Deepgram succeeds
        mock_post.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            Exception("fail 3"),
            deepgram_mock,             # Deepgram fallback call
        ]
        stt = STTProvider()
        result = stt.transcribe(DUMMY_AUDIO)

    assert result == "deepgram after retries"
    log.info("PASS | test_stt_falls_back_after_all_retries_exhausted")

