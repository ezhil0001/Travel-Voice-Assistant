import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

from config import settings

# Two things live here:
#
#   1. BaseSTTProvider / BaseTTSProvider — abstract interfaces that every
#      provider class must implement. This mirrors the TypeScript reference
#      architecture where each provider (Sarvam, Deepgram) was a separate
#      class behind a shared `LLMServiceIntf`. Defining the interface here
#      means adding a third provider later only requires a new class — the
#      orchestrators (STTProvider / TTSProvider) never need to change.
#
#   2. retry_call() — shared exponential-backoff utility used by both STT
#      and TTS orchestrators. Centralising it here (rather than duplicating
#      inside each provider, as the TS reference did) keeps provider classes
#      focused solely on their HTTP contract.

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Attempt 0 → immediate, Attempt 1 → 2 s, Attempt 2 → 4 s
_BACKOFF_FACTOR = 2.0


class BaseSTTProvider(ABC):
    """Common interface for all Speech-to-Text providers.

    Each concrete provider (Sarvam, Deepgram, …) implements only `request()` —
    the single raw HTTP call for that provider. Retry logic and fallback
    orchestration live in the STTProvider orchestrator, not here.

    This mirrors the TypeScript pattern:
        DeepgramStreamSTTService implements LLMServiceIntf
        SarvamBatchService      implements LLMServiceIntf
    """

    @abstractmethod
    def request(self, audio_bytes: bytes) -> str:
        """Make one transcription attempt. Must raise on any failure.

        Args:
            audio_bytes: Raw audio data (WAV or MP3).

        Returns:
            Transcribed text string.

        Raises:
            Any exception on HTTP error, parse failure, or timeout.
        """
        ...


class BaseTTSProvider(ABC):
    """Common interface for all Text-to-Speech providers.

    Each concrete provider (Sarvam, Deepgram, …) implements only `request()`.
    Retry and fallback orchestration live in the TTSProvider orchestrator.
    """

    @abstractmethod
    def request(self, text: str) -> bytes:
        """Make one synthesis attempt. Must raise on any failure.

        Args:
            text: Clean, markdown-free text ready for speech synthesis.

        Returns:
            Raw audio bytes (WAV or MP3).

        Raises:
            Any exception on HTTP error, parse failure, or timeout.
        """
        ...


def retry_call(
    fn: Callable[[], T],
    label: str,
    max_attempts: int | None = None,
) -> T:
    """Call fn() up to max_attempts times with exponential backoff.

    Args:
        fn:           Zero-argument callable wrapping a single provider request.
        label:        Short identifier for log messages (e.g. "STT/Sarvam").
        max_attempts: Total attempts allowed. Defaults to VOICE_MAX_RETRIES + 1.

    Returns:
        The return value of fn() on the first successful call.

    Raises:
        The last exception raised by fn() after all attempts are exhausted.
    """
    if max_attempts is None:
        max_attempts = int(settings.VOICE_MAX_RETRIES) + 1

    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        if attempt > 0:
            wait = _BACKOFF_FACTOR * (2 ** (attempt - 1))
            logger.info(
                "%s | retry %d/%d — waiting %.0f s",
                label, attempt, max_attempts - 1, wait,
            )
            time.sleep(wait)

        try:
            result = fn()
            logger.info("%s | succeeded on attempt %d", label, attempt + 1)
            return result
        except Exception as exc:
            logger.warning("%s | attempt %d failed: %s", label, attempt + 1, exc)
            last_exc = exc

    raise last_exc  # type: ignore[misc]
