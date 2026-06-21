# Text-to-speech orchestrator.
#
# Architecture adapted from the TypeScript reference:
#   - Each provider (Sarvam, Deepgram) is an independent class implementing
#     BaseTTSProvider — mirrors the TS pattern where each service was a
#     separate class behind a shared interface.
#   - TTSProvider is the orchestrator: it owns priority order and wires
#     retry_call() around the primary, then falls through to the fallback.
#     It does NOT contain any HTTP logic — that belongs to each provider class.
#   - Adding a third TTS provider requires only a new class. TTSProvider
#     receives it via the constructor and needs no internal changes.

import base64
import logging
import os

import requests

from config import settings
from voice.base import BaseTTSProvider, retry_call

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "test_results.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ── Provider implementations ──────────────────────────────────────────────────

class SarvamTTS(BaseTTSProvider):
    """Sarvam AI Text-to-Speech provider.

    Handles the Sarvam-specific JSON request and base64 audio decoding.
    Raises on any failure so retry_call() and the orchestrator can react.
    """

    _URL = "https://api.sarvam.ai/text-to-speech"

    def request(self, text: str) -> bytes:
        logger.info("TTS | SarvamTTS.request — synthesising %d chars", len(text))
        headers = {
            "api-subscription-key": settings.SARVAM_API_KEY,
            "Content-Type": "application/json",
        }
        body = {
            "inputs": [text],
            "target_language_code": "en-IN",
            "speaker": "meera",
            "model": "bulbul:v1",
        }

        response = requests.post(self._URL, headers=headers, json=body, timeout=20)
        response.raise_for_status()
        # Sarvam returns audio as base64 inside JSON — decode before returning
        audio_bytes = base64.b64decode(response.json()["audios"][0])
        logger.info("TTS | SarvamTTS returned %d bytes", len(audio_bytes))
        return audio_bytes


class DeepgramTTS(BaseTTSProvider):
    """Deepgram Text-to-Speech provider.

    Handles Deepgram's streaming TTS endpoint. Audio bytes are returned
    directly in the response body — no decoding step needed.
    Raises on any failure so the orchestrator treats it as a fallback signal.
    """

    _URL = "https://api.deepgram.com/v1/speak"

    def request(self, text: str) -> bytes:
        logger.info("TTS | DeepgramTTS.request — synthesising %d chars", len(text))
        headers = {
            "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
            "Content-Type": "application/json",
        }
        params = {"model": "aura-asteria-en"}
        body   = {"text": text}

        response = requests.post(self._URL, headers=headers, params=params, json=body, timeout=20)
        response.raise_for_status()
        logger.info("TTS | DeepgramTTS returned %d bytes", len(response.content))
        return response.content


# ── Orchestrator ───────────────────────────────────────────────────────────────

class TTSProvider:
    """TTS orchestrator that applies retry + fallback across registered providers.

    Resilience design (adapted from TS reference):
        1. retry_call() wraps the primary provider in an exponential-backoff loop
           (VOICE_MAX_RETRIES attempts). Transient failures resolve here silently.
        2. If all primary retries fail, falls through to the fallback provider
           (tried once — it's the last resort).
        3. If the fallback also fails, returns b"" so the caller always gets bytes.

    The orchestrator owns the priority order. Providers own their HTTP logic.
    """

    def __init__(
        self,
        primary: BaseTTSProvider | None = None,
        fallback: BaseTTSProvider | None = None,
    ) -> None:
        # Default chain: Sarvam (primary) → Deepgram (fallback).
        # Tests can inject mocked providers directly via these constructor args.
        self._primary  = primary  or SarvamTTS()
        self._fallback = fallback or DeepgramTTS()

    def synthesize(self, text: str) -> bytes:
        """Convert plain text to audio bytes with automatic retry and fallback.

        Args:
            text: Clean, markdown-free text from PostModelMiddleware.

        Returns:
            Raw audio bytes (WAV or MP3), or b"" if all providers fail.
        """
        # --- Primary with retry ---
        try:
            return retry_call(
                fn=lambda: self._primary.request(text),
                label="TTS/Sarvam",
            )
        except Exception:
            logger.warning("TTS | primary exhausted all retries — switching to fallback")

        # --- Fallback, tried once ---
        try:
            return retry_call(
                fn=lambda: self._fallback.request(text),
                label="TTS/Deepgram",
                max_attempts=1,
            )
        except Exception:
            logger.error("TTS | all providers failed — returning empty bytes")
            return b""


