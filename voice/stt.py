# Speech-to-text orchestrator.
#
# Two providers are registered: Sarvam (primary) and Deepgram (fallback).
# STTProvider applies retry_call() around the primary first — Sarvam can be
# slow to respond after a period of inactivity, and most failures resolve
# within the retry window without needing to switch providers.
# Deepgram is only called after all primary retries are exhausted.
#
# Each provider class owns its own HTTP logic. STTProvider owns the
# priority order and resilience wiring. Adding a third provider means
# adding a new class and updating the constructor — nothing else changes.

import io
import logging
import os

import requests

from config import settings
from voice.base import BaseSTTProvider, retry_call

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "test_results.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ── Provider implementations ──────────────────────────────────────────────────

class SarvamSTT(BaseSTTProvider):
    """Sarvam AI Speech-to-Text provider.

    Handles the Sarvam-specific multipart form upload and transcript parsing.
    Raises on any failure so retry_call() and the orchestrator can react.
    """

    _URL = "https://api.sarvam.ai/speech-to-text"

    def request(self, audio_bytes: bytes) -> str:
        logger.info("STT | SarvamSTT.request — sending audio (%d bytes)", len(audio_bytes))

        # Detect the real audio container from the magic bytes.
        # MediaRecorder in Chrome/Firefox produces WebM (starts with \x1a\x45\xdf\xa3)
        # even when the code names the blob "audio.wav".  Sarvam accepts WebM
        # natively, so we send the correct MIME type so the API parses it properly.
        if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
            filename  = "audio.webm"
            mime_type = "audio/webm"
        elif audio_bytes[:4] == b"RIFF":
            filename  = "audio.wav"
            mime_type = "audio/wav"
        else:
            # Fallback — send as-is and let Sarvam auto-detect
            filename  = "audio.webm"
            mime_type = "audio/webm"

        files   = {"file": (filename, io.BytesIO(audio_bytes), mime_type)}
        data    = {
            "model":         "saaras:v3",    # saarika:v2 is deprecated → 400
            "mode":          "transcribe",   # required for saaras:v3
            "language_code": "en-IN",
        }
        headers = {"api-subscription-key": settings.SARVAM_API_KEY}

        try:
            response = requests.post(self._URL, headers=headers, files=files, data=data, timeout=30)
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "STT | SarvamSTT HTTP %s — body: %s",
                exc.response.status_code, exc.response.text[:500],
            )
            raise

        transcript = response.json().get("transcript", "")
        logger.info("STT | SarvamSTT returned %d chars", len(transcript))
        return transcript


class DeepgramSTT(BaseSTTProvider):
    """Deepgram Speech-to-Text provider.

    Handles Deepgram's raw-bytes POST and nested JSON transcript extraction.
    Raises on any failure so the orchestrator can treat it as a fallback signal.
    """

    _URL = "https://api.deepgram.com/v1/listen"

    def request(self, audio_bytes: bytes) -> str:
        logger.info("STT | DeepgramSTT.request — sending audio (%d bytes)", len(audio_bytes))
        headers = {"Authorization": f"Token {settings.DEEPGRAM_API_KEY}"}
        params  = {"punctuate": "true", "language": "en"}

        response = requests.post(
            self._URL, headers=headers, params=params, data=audio_bytes, timeout=15
        )
        response.raise_for_status()
        transcript = (
            response.json()
            ["results"]["channels"][0]["alternatives"][0]["transcript"]
        )
        logger.info("STT | DeepgramSTT returned %d chars", len(transcript))
        return transcript


# ── Orchestrator ───────────────────────────────────────────────────────────────

class STTProvider:
    """STT orchestrator that applies retry + fallback across registered providers.

    Sarvam is tried first with exponential backoff (VOICE_MAX_RETRIES attempts).
    If all retries fail, Deepgram is tried once as a last resort. If Deepgram
    also fails, an empty string is returned so the caller always gets a string.

    The orchestrator owns the priority order. Providers own their HTTP logic.
    """

    def __init__(
        self,
        primary: BaseSTTProvider | None = None,
        fallback: BaseSTTProvider | None = None,
    ) -> None:
        # Default chain: Sarvam (primary) → Deepgram (fallback).
        # Tests can inject mocked providers directly via these constructor args.
        self._primary  = primary  or SarvamSTT()
        self._fallback = fallback or DeepgramSTT()

    def transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio bytes to text with automatic retry and fallback.

        Args:
            audio_bytes: Raw audio data from the voice input layer.

        Returns:
            Transcribed string, or "" if all providers fail.
        """
        # --- Primary with retry ---
        try:
            return retry_call(
                fn=lambda: self._primary.request(audio_bytes),
                label="STT/Sarvam",
            )
        except Exception:
            logger.warning("STT | primary exhausted all retries — switching to fallback")

        # --- Fallback, tried once ---
        try:
            return retry_call(
                fn=lambda: self._fallback.request(audio_bytes),
                label="STT/Deepgram",
                max_attempts=1,
            )
        except Exception:
            logger.error("STT | all providers failed — returning empty string")
            return ""


