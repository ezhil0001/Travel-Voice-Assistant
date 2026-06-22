import re
from typing import Optional

from middleware.base import AgentMiddleware
from config import settings

# Implements the after_model hook — intercepts the LLM response before it
# reaches the TTS layer. Strips markdown and URLs that TTS would either
# mispronounce or read out as noise, then trims replies that are too long.
# Both the char limit and the overflow message are read from settings so they
# can be tuned via .env without touching this file.


class PostModelMiddleware(AgentMiddleware):
    """Sanitises LLM output before it reaches the TTS layer.

    All tunable values (max response length, overflow suffix) are injected
    from config/settings.py — nothing is hardcoded in this class.
    """

    def __init__(self) -> None:
        self._max_chars: int = settings.POST_MODEL_MAX_CHARS
        self._overflow_suffix: str = settings.POST_MODEL_OVERFLOW_SUFFIX

    def after_model(self, state: dict, **kwargs) -> Optional[dict]:
        """Strip markdown/URLs and truncate if the response is too long.

        Reads 'raw_response' from state and writes 'cleaned_response' back.
        """
        raw = state.get("raw_response", "")
        state["cleaned_response"] = self.process(raw)
        return state

    # ------------------------------------------------------------------
    # Direct-call wrapper — keeps tests readable and graph nodes simple
    # ------------------------------------------------------------------

    def process(self, raw_response: str) -> str:
        """Strip markdown and URLs, then truncate if the text is too long."""
        text = self._strip_markdown(raw_response)
        text = self._strip_urls(text)
        text = self._truncate(text)
        return text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _strip_markdown(self, text: str) -> str:
        """Remove common markdown tokens that TTS reads literally.

        Bold (**word**), italic (*word*), headers (## heading), and
        bullet dashes are all collapsed to plain text.
        """
        text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)   # **bold** / *italic*
        text = re.sub(r"#{1,6}\s*", "", text)                  # ## headers
        text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)    # bullet points
        text = re.sub(r"`{1,3}.*?`{1,3}", "", text)            # inline / fenced code
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) → label
        return text.strip()

    def _strip_urls(self, text: str) -> str:
        """Remove bare URLs — they sound terrible when read aloud."""
        return re.sub(r"https?://\S+", "", text).strip()

    def _truncate(self, text: str) -> str:
        """Cap response length at POST_MODEL_MAX_CHARS (from .env).

        Cuts at the last sentence boundary before the limit so the response
        never ends mid-word or mid-thought. Falls back to a hard character cut
        only when no sentence boundary exists before the limit.
        """
        if len(text) <= self._max_chars:
            return text
        # Find the last sentence-ending punctuation before the char limit
        window = text[:self._max_chars]
        last_boundary = max(
            window.rfind(". "),
            window.rfind("? "),
            window.rfind("! "),
            window.rfind(".\n"),
        )
        if last_boundary > self._max_chars // 2:
            # Cut at the sentence end (include the punctuation, skip the space)
            return text[:last_boundary + 1].rstrip() + " " + self._overflow_suffix
        # No clean boundary found — hard cut at word boundary
        return window.rstrip() + " " + self._overflow_suffix

