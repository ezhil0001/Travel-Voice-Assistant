import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

from config import settings

# Sarvam is the primary provider for both STT and TTS. It can return 5xx errors
# during cold-start or peak load — these failures are transient and resolve within
# a few seconds. Retrying with backoff avoids unnecessary provider switches
# and keeps Deepgram usage reserved for genuine outages rather than blips.
#
# retry_call() is centralised here rather than duplicated inside each provider
# so the retry behaviour is consistent and only needs to change in one place.

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Attempt 0 → immediate, Attempt 1 → 2 s, Attempt 2 → 4 s
_BACKOFF_FACTOR = 2.0


class BaseSTTProvider(ABC):
    """Interface every speech-to-text provider must implement.

    Keeping implementations behind a shared interface means the STTProvider
    orchestrator can switch between Sarvam and Deepgram without changing its
    own logic — it just holds a reference to whichever provider it is trying.
    """

    @abstractmethod
    def request(self, audio_bytes: bytes) -> str:
        """Attempt a single transcription call. Must raise on any failure.

        Args:
            audio_bytes: Raw audio data (WAV or MP3).

        Returns:
            Transcribed text string.

        Raises:
            Any exception on HTTP error, parse failure, or timeout.
        """
        ...


class BaseTTSProvider(ABC):
    """Interface every text-to-speech provider must implement.

    The TTSProvider orchestrator calls request() through retry_call() and
    switches to Deepgram only after all Sarvam retries are exhausted.
    """

    @abstractmethod
    def request(self, text: str) -> bytes:
        """Attempt a single synthesis call. Must raise on any failure.

        Args:
            text: Clean, markdown-free text ready for synthesis.

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
    """Call fn() up to max_attempts times, backing off between failures.

    The backoff window gives Sarvam time to recover from a momentary spike
    before the caller decides it is genuinely down and switches to Deepgram.

    Args:
        fn:           Zero-argument callable wrapping one provider call.
        label:        Short identifier for log messages (e.g. "STT/Sarvam").
        max_attempts: Total attempts including the first.
                      Defaults to VOICE_MAX_RETRIES + 1 from .env.

    Returns:
        The return value of fn() on a successful call.

    Raises:
        The last exception raised after all attempts fail.
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
