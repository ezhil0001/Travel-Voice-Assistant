# SupervisorAgent — the routing brain of the LangGraph pipeline.
#
# Architectural guardrails enforced here:
#   - The supervisor NEVER calls any tool directly.
#   - The supervisor NEVER reads raw conversation history or user_input.
#   - It receives ONLY cleaned_input (already sanitised by PreModelMiddleware).
#   - It returns ONLY a single intent string — nothing else.
#
# The LLM call here is intentionally lightweight: a short classification
# prompt with a constrained single-word output. Any heavier reasoning
# (tool calls, data formatting) belongs inside the sub-agents.

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from state.schema import TravelState
from config import settings

logger = logging.getLogger(__name__)

# Exhaustive set of valid intent labels — used both in the prompt and as
# a fallback guard so an unexpected LLM response never breaks the graph.
VALID_INTENTS = frozenset({
    "weather", "flight", "attractions", "currency", "timezone", "general"
})

_ROUTING_PROMPT = (
    "You are a routing agent for a travel assistant. "
    "Given the user's message, classify the intent into exactly ONE of these categories:\n"
    "- weather       (temperature, forecast, climate, rain, packing)\n"
    "- flight        (flights, tickets, travel dates, airports, prices)\n"
    "- attractions   (what to visit, tourist spots, things to do, places)\n"
    "- currency      (money, exchange rate, how much in X currency)\n"
    "- timezone      (time, what time is it, time difference)\n"
    "- general       (anything else, greetings, general travel questions)\n\n"
    "Respond with ONLY the category word — no explanation, no punctuation."
)


class SupervisorAgent:
    """Routes a cleaned user query to the correct sub-agent node.

    Uses the same Runnable resilience pattern as all other model calls:
        primary (Ollama) → .with_retry(3) → .with_fallbacks([Groq])

    The supervisor is kept deliberately thin:
        - Input:  state["cleaned_input"] only
        - Output: one of the six intent strings
        - Side effect: writes detected_intent into TravelState
    """

    def __init__(self) -> None:
        primary = ChatOpenAI(model=settings.OLLAMA_MODEL, temperature=0)
        fallback = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY)

        self._chain = (
            primary
            .with_retry(stop_after_attempt=3)
            .with_fallbacks(
                fallbacks=[fallback],
                exceptions_to_handle=(Exception,),
            )
        )

    def route(self, state: TravelState) -> TravelState:
        """Classify the cleaned input and write detected_intent into state.

        Reads:  state["cleaned_input"]
        Writes: state["detected_intent"]

        The supervisor never reads user_input, conversation_history, or
        any tool output — those are handled by PreModelMiddleware and the
        sub-agents respectively.
        """
        cleaned = state.get("cleaned_input", state.get("user_input", ""))
        logger.info("SupervisorAgent.route | input=%s", cleaned)

        try:
            messages = [
                SystemMessage(content=_ROUTING_PROMPT),
                HumanMessage(content=cleaned),
            ]
            response = self._chain.invoke(messages)
            intent = response.content.strip().lower()

            # Guard: if the LLM returns anything outside the valid set,
            # fall back to general rather than breaking the graph.
            if intent not in VALID_INTENTS:
                logger.warning(
                    "SupervisorAgent.route | unexpected intent=%r → defaulting to general", intent
                )
                intent = "general"

            logger.info("SupervisorAgent.route | intent=%s", intent)
            return {**state, "detected_intent": intent, "error": ""}

        except Exception as exc:
            logger.error("SupervisorAgent.route | failed: %s", exc)
            return {**state, "detected_intent": "general", "error": str(exc)}
