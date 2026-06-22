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

# Location pronouns that refer to a previously-mentioned destination.
# When any of these appear in a query that has no explicit city, they are
# replaced with the last city found in conversation history.
# Examples: "What time is it there?" → "What time is it in Tokyo?"
#           "Is that city expensive?" → "Is Tokyo expensive?"
_LOCATION_PRONOUN_RE = re.compile(
    r'\b(there|that city|that place|that destination|the city|'
    r'the destination|the place|that country|the country|the location)\b',
    re.IGNORECASE,
)


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
        # Resolve pronouns ("there", "that city") using history BEFORE city detection
        cleaned = self._resolve_location_references(cleaned, history)
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

    def _resolve_location_references(self, text: str, history: list) -> str:
        """Replace location pronouns with the last known city from history.

        When a user says "What time is it there?" after discussing Tokyo, this
        substitutes "there" with "Tokyo" so every downstream component (supervisor,
        agents, validators) sees an unambiguous city name rather than a pronoun.

        Only triggers when:
          - The query contains a location pronoun (there, that city, etc.)
          - The query itself has NO city name already (avoids double substitution)
          - A city can be found in recent conversation history

        Examples:
          "What time is it there?"       → "What time is it in Tokyo?"
          "Is that city expensive?"      → "Is Tokyo expensive?"
          "How's the weather in Paris?"  → unchanged (city already explicit)
        """
        if not _LOCATION_PRONOUN_RE.search(text):
            return text
        # If the query already has an explicit city, no substitution needed
        if self._detect_city(text):
            return text
        last_city = self._last_city_from_history(history)
        if not last_city:
            return text

        def _replacer(m: re.Match) -> str:
            # Keep grammatical context: "there" → "in <City>", noun phrases → "<City>"
            token = m.group(1).lower()
            if token == "there":
                return f"in {last_city}"
            return last_city

        resolved = _LOCATION_PRONOUN_RE.sub(_replacer, text)
        return resolved

    def _last_city_from_history(self, history: list) -> str | None:
        """Scan conversation history (most-recent-first) for the last city mentioned.

        Searches both user and assistant turns so it works regardless of which
        side said the city name most recently.
        """
        for turn in reversed(history):
            content = turn.get("content", "")
            city = self._detect_city(content)
            if city:
                return city
        return None

    def _last_n_turns(self, history: list) -> list:
        """Slice the last N turns from conversation history.

        N is controlled by PRE_MODEL_HISTORY_WINDOW in .env — increasing it
        gives the LLM more context at the cost of a slightly longer prompt.
        """
        return history[-self._history_window:] if history else []
