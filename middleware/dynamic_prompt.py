import json
import os
from typing import Any, Callable, Optional

from middleware.base import AgentMiddleware

# Implements the wrap_model_call hook — intercepts the request just before
# it reaches the LLM and injects a fully assembled, context-aware system prompt.
# All prompt text lives in config/prompts.json — wording changes never require
# touching this file.

_PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "prompts.json")

with open(_PROMPTS_PATH, encoding="utf-8") as _f:
    _PROMPTS: dict = json.load(_f)


class DynamicPromptBuilder(AgentMiddleware):
    """Assembles a context-aware system prompt and injects it into the model request.

    All text content (base prompt, per-agent instructions, IATA constraint) is
    loaded from config/prompts.json at startup. The class contains zero
    hardcoded strings — update prompts.json to change behaviour in any environment.

    Prompt layers:
        1. Shared base — tone, verbosity, and voice-first constraints.
        2. Agent-specific instruction — domain guidance per active sub-agent.
        3. Recent conversation context — last few turns for pronoun resolution.
    """

    def __init__(self) -> None:
        self._base: str = _PROMPTS["base_system_prompt"]
        self._instructions: dict[str, str] = _PROMPTS["agent_instructions"]
        self._default_type: str = _PROMPTS["default_agent_type"]
        self._iata_constraint: str = _PROMPTS["flight_iata_constraint"]

    def wrap_model_call(
        self,
        request: dict,
        handler: Callable[[dict], Any],
        **kwargs,
    ) -> Any:
        """Inject the assembled prompt into the request before calling the model.

        Reads 'agent_type' and 'context' from request, builds the system prompt,
        then forwards the enriched request to the next handler in the chain.
        """
        agent_type = request.get("agent_type", self._default_type)
        context = request.get("context", {})
        cleaned_input = request.get("cleaned_input", "")

        request["prompt"] = self.build(cleaned_input, context, agent_type)
        return handler(request)

    # ------------------------------------------------------------------
    # Direct-call wrapper — used by tests and graph nodes that build prompts
    # without going through the full pipeline
    # ------------------------------------------------------------------

    def build(self, cleaned_input: str, context: dict, agent_type: str) -> str:
        """Return a fully assembled system prompt string.

        Args:
            cleaned_input: The sanitised user message (from PreModelMiddleware).
            context:       Dict that may contain a 'history' key with recent turns.
            agent_type:    Agent domain key — must match a key in prompts.json
                           agent_instructions. Unrecognised types fall back to
                           the default_agent_type defined in prompts.json.

        Returns:
            A single string ready to be passed as the system message to the LLM.
        """
        parts: list[str] = [self._base]

        # Fall back to the configured default if the agent type isn't recognised.
        instruction = self._instructions.get(
            agent_type, self._instructions[self._default_type]
        )
        parts.append(f"\nCurrent task context: {instruction}")

        # The IATA constraint is only relevant for the flight agent — it's stored
        # in prompts.json so the wording can be tuned without code changes.
        if agent_type == "flight":
            parts.append(f"\n{self._iata_constraint}")

        # Surface recent turns so the LLM can resolve pronouns like 'there' or
        # 'it' without asking the user to repeat the destination.
        history: list = context.get("history", [])
        if history:
            parts.append("\nRecent conversation:")
            for turn in history:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "")
                parts.append(f"  {role}: {content}")

        return "\n".join(parts)
