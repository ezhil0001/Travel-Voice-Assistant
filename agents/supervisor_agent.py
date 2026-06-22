# SupervisorAgent — detects ALL intents in a user query and returns them as a list.
#
# Intent detection is handled entirely by the LLM. A well-constructed prompt
# with clear examples covers semantic meaning — "how will it drain my wallet"
# correctly maps to "currency" the same way "exchange rate" does.
# Keyword matching can only catch exact words and misses paraphrases entirely,
# so the LLM is the only detection layer here.
#
# If the LLM is unavailable or returns unparseable output, the fallback is
# ["general"] so the graph always has a safe path to continue with.
#
# The supervisor never calls tools and never reads raw conversation history.
# It receives only cleaned_input and writes only detected_intents.

import json
import logging
import os
import re

from agents.model_layer import ModelLayer
from state.schema import TravelState

logger = logging.getLogger(__name__)

_PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "prompts.json")
with open(_PROMPTS_PATH, encoding="utf-8") as _f:
    _PROMPTS: dict = json.load(_f)

_ROUTING_PROMPT_TEMPLATE: str = _PROMPTS["supervisor_routing_prompt"]

# VALID_INTENTS is derived from agent_instructions keys in prompts.json so
# adding a new domain only requires updating config — not this file.
VALID_INTENTS: frozenset[str] = frozenset(_PROMPTS["agent_instructions"].keys())


class SupervisorAgent:
    """Routes a cleaned user query to one or more sub-agent domains.

    Returns a list so a compound query ("weather + currency + attractions")
    triggers all three agents. Detection is entirely LLM-based — no keyword
    matching. If the LLM call fails or returns unparseable output, returns
    ["general"] as a safe fallback so the graph never stalls.
    """

    def __init__(self) -> None:
        self._model = ModelLayer()

    def route(self, state: TravelState) -> TravelState:
        """Detect all intents and write them into state["detected_intents"].

        Reads:  state["cleaned_input"]
        Writes: state["detected_intents"], state["error"]
        """
        cleaned = state.get("cleaned_input") or state.get("user_input", "")
        logger.info("SupervisorAgent.route | input=%s", cleaned[:80])

        intents = self._llm_detect(cleaned)
        logger.info("SupervisorAgent.route | intents=%s", intents)
        return {**state, "detected_intents": intents, "error": ""}

    def _llm_detect(self, user_input: str) -> list[str]:
        """Ask the LLM for all intents present in the query.

        Returns a validated, deduplicated list. Returns ["general"] on any
        failure — parse error, timeout, or LLM returning nothing useful.
        """
        try:
            prompt = _ROUTING_PROMPT_TEMPLATE.format(user_input=user_input)
            raw = self._model.invoke(prompt)
            logger.info("SupervisorAgent | LLM raw=%s", raw[:200])

            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not match:
                logger.warning("SupervisorAgent | no JSON array in LLM output — returning general")
                return ["general"]

            parsed = json.loads(match.group())
            validated = [
                i.strip().lower()
                for i in parsed
                if isinstance(i, str) and i.strip().lower() in VALID_INTENTS
            ]
            if not validated:
                logger.warning("SupervisorAgent | no valid intents in %s — returning general", parsed)
                return ["general"]

            return _deduplicate(validated)

        except Exception as exc:
            logger.error("SupervisorAgent | LLM detection failed: %s — returning general", exc)
            return ["general"]


def _deduplicate(items: list[str]) -> list[str]:
    """Remove duplicates while preserving insertion order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


