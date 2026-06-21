# Speech-to-text orchestrator.
#
# Architecture adapted from the TypeScript reference:
#   - Each provider (Sarvam, Deepgram) is an independent class implementing
#     BaseSTTProvider — same pattern as Sarvam/DeepgramStreamSTTService each
#     implementing LLMServiceIntf in the TS codebase.
#   - STTProvider is the orchestrator: it knows the priority order and wires
#     retry_call() around the primary, then falls through to the fallback.
#     It does NOT know about HTTP — that's each provider's responsibility.
#   - Adding a third provider (e.g. AssemblyAI) is a new class only —
#     STTProvider just adds it to the chain. Nothing else changes.

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
        files   = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}
        data    = {"language_code": "en-IN", "model": "saarika:v2"}
        headers = {"api-subscription-key": settings.SARVAM_API_KEY}

        response = requests.post(self._URL, headers=headers, files=files, data=data, timeout=15)
        response.raise_for_status()
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

    Resilience design (adapted from TS reference):
        1. retry_call() wraps the primary provider in an exponential-backoff loop
           (VOICE_MAX_RETRIES attempts). Transient failures resolve here silently.
        2. If all primary retries fail, falls through to the fallback provider
           (tried once — it's the last resort).
        3. If the fallback also fails, returns "" so the caller always gets a string.

    The orchestrator owns the priority order. Providers own their HTTP logic.
    This separation means swapping or adding providers requires zero changes here.
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


