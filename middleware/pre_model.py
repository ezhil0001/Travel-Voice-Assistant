import json
import os
import re
from datetime import datetime
from typing import Optional

from middleware.base import AgentMiddleware
from config import settings

# Implements the before_model hook — intercepts agent state before the LLM sees it.
# Voice transcriptions often carry extra whitespace, repeated words, or
# inconsistent capitalisation — this hook normalises all of that so every
# downstream component works with clean, predictable text.

# Load the city list from config/cities.json at module level so the sorted
# lookup set is built once and reused across every request.
_CITIES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "cities.json")

with open(_CITIES_PATH, encoding="utf-8") as _f:
    _raw_cities: list[str] = json.load(_f)

# Sort by descending length once so multi-word cities ("new york") always
# match before shorter substrings ("york") — no re-sorting on every call.
_KNOWN_CITIES_SORTED: list[str] = sorted(_raw_cities, key=len, reverse=True)


class PreModelMiddleware(AgentMiddleware):
    """Sanitises raw STT output and enriches state with conversation context
    before the string reaches the LLM.

    All tunable values (history window size, city list) are loaded from
    config — nothing is hardcoded in this class.
    """

    def __init__(self) -> None:
        # Read the window size from settings so it can be changed via .env
        # without touching any Python source files.
        self._history_window: int = settings.PRE_MODEL_HISTORY_WINDOW

    def before_model(self, state: dict, **kwargs) -> Optional[dict]:
        """Clean the transcribed input and attach context metadata.

        Reads 'user_input' and 'conversation_history' from state and writes
        back 'cleaned_input', 'context', and 'metadata'.
        """
        raw = state.get("user_input", "")
        history = state.get("conversation_history", [])

        cleaned = self._clean(raw)
        city = self._detect_city(cleaned)
        context = self._last_n_turns(history)

        state["cleaned_input"] = cleaned
        state["context"] = context
        state["metadata"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_length": len(cleaned),
            "detected_city": city,
        }
        return state

    # ------------------------------------------------------------------
    # Direct-call wrapper — keeps tests and graph nodes simple
    # ------------------------------------------------------------------

    def process(self, user_input: str, conversation_history: list) -> dict:
        """Convenience wrapper so callers don't have to build a state dict manually."""
        state = {
            "user_input": user_input,
            "conversation_history": conversation_history,
        }
        result = self.before_model(state)
        return {
            "cleaned_input": result["cleaned_input"],
            "context": result["context"],
            "metadata": result["metadata"],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        """Collapse repeated whitespace and strip leading/trailing spaces.

        STT engines sometimes produce multiple consecutive spaces when the
        speaker pauses mid-sentence — a single regex pass fixes all of them.
        """
        text = text.strip()
        text = re.sub(r" {2,}", " ", text)  # collapse runs of spaces
        text = re.sub(r"\n+", " ", text)    # flatten newlines from some STT outputs
        return text

    def _detect_city(self, text: str) -> str | None:
        """Return the first matching city name found in the text, or None.

        Uses the pre-sorted list from cities.json — longest names checked first
        so 'New York' matches before 'York'. The city list is owned entirely by
        config/cities.json and requires no code changes to extend.
        """
        lower = text.lower()
        for city in _KNOWN_CITIES_SORTED:
            if city in lower:
                return city.title()
        return None

    def _last_n_turns(self, history: list) -> list:
        """Slice the last N turns from conversation history.

        N is controlled by PRE_MODEL_HISTORY_WINDOW in .env — increasing it
        gives the LLM more context at the cost of a slightly longer prompt.
        """
        return history[-self._history_window:] if history else []
