# Text-to-speech orchestrator.
#
# Two providers are registered: Sarvam (primary) and Deepgram (fallback).
# TTSProvider applies retry_call() around the primary first — Sarvam cold-start
# latency can cause the first request in a quiet period to time out, and
# retries with backoff resolve the majority of those cases before falling over
# to Deepgram.
#
# Each provider class owns its own HTTP logic. TTSProvider owns the
# priority order and resilience wiring. Adding a third provider means
# adding a new class and updating the constructor — nothing else changes.

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
            "text":                 text,         # bulbul:v3 uses "text", not "inputs"
            "target_language_code": "en-IN",
            "speaker":              "ritu",        # valid bulbul:v3 speaker (arya not in v3 list)
            "model":                "bulbul:v3",   # bulbul:v1 deprecated → 400
        }

        try:
            response = requests.post(self._URL, headers=headers, json=body, timeout=20)
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "TTS | SarvamTTS HTTP %s — body: %s",
                exc.response.status_code, exc.response.text[:500],
            )
            raise

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

    Sarvam is tried first with exponential backoff (VOICE_MAX_RETRIES attempts).
    If all retries fail, Deepgram is tried once as a last resort. If Deepgram
    also fails, empty bytes are returned so the caller always gets bytes.

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


