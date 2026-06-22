# LangGraph StateGraph — dynamic multi-intent travel assistant pipeline.
#
# Graph topology (always linear — no conditional branching needed):
#   START → pre_middleware → supervisor → run_agents → merge_responses → post_middleware → END
#
# Key design decisions:
#   - supervisor returns a LIST of intents, not a single string.
#   - run_agents iterates over all detected intents, running every matching agent.
#     One intent or five intents — the same node handles both with no special cases.
#   - merge_responses calls the LLM only when more than one agent responded.
#     Single-intent queries bypass the merge step to avoid an unnecessary LLM call.
#   - All prompt text (merge template, focused-query template) is loaded from
#     config/prompts.json — nothing is hardcoded in this file.

import json
import logging
import os
import time
from datetime import date as _date

from langgraph.graph import StateGraph, START, END

from agents.supervisor_agent import SupervisorAgent
from agents.weather_agent import WeatherAgent
from agents.flight_agent import FlightAgent
from agents.attractions_agent import AttractionsAgent
from agents.currency_agent import CurrencyAgent
from agents.timezone_agent import TimezoneAgent
from agents.model_layer import ModelLayer
from middleware.pre_model import PreModelMiddleware
from middleware.post_model import PostModelMiddleware
from state.schema import TravelState

logger = logging.getLogger(__name__)

# ── Prompt templates loaded from config ───────────────────────────────────────

_PROMPTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "prompts.json")
with open(_PROMPTS_PATH, encoding="utf-8") as _f:
    _PROMPTS: dict = json.load(_f)

_MERGE_TEMPLATE: str = _PROMPTS["merge_responses_prompt"]
_BATCH_FOCUSED_TEMPLATE: str = _PROMPTS["batch_focused_query_prompt"]
_SUMMARY_TEMPLATE: str = _PROMPTS["summary_prompt"]

# Python-level lookup maps — used to validate/fix LLM output for flight and
# timezone queries. Loaded from prompts.json so they can be extended without
# touching this file.
_CITY_TO_IATA: dict[str, str]     = {k.lower(): v for k, v in _PROMPTS.get("city_to_iata", {}).items()}
_CITY_TO_TZ:   dict[str, str]     = {k.lower(): v for k, v in _PROMPTS.get("city_to_timezone", {}).items()}

# Human-readable labels used in tool_events (loaded from config agent_instructions keys)
_INTENT_LABELS: dict[str, str] = {
    "weather":     "Fetching weather data",
    "flight":      "Searching flights",
    "attractions": "Finding attractions and itinerary",
    "currency":    "Converting currency",
    "timezone":    "Checking time zone",
    "general":     "General travel advice",
}

# Human-readable data source names surfaced in the Tool Activity panel
_INTENT_SOURCES: dict[str, str] = {
    "weather":     "OpenWeatherMap API",
    "flight":      "FlightAPI.io",
    "attractions": "GeoNames API",
    "currency":    "ExchangeRate-API",
    "timezone":    "TimeAPI.io",
    "general":     "LLM Knowledge",
}

# ── Singleton instances — built once, reused across all graph invocations ──────

_supervisor        = SupervisorAgent()
_pre_mw            = PreModelMiddleware()
_post_mw           = PostModelMiddleware()

# Agent registry — maps intent string → agent instance.
# Add a new intent here and nowhere else; the rest of the graph picks it up automatically.
_AGENT_REGISTRY: dict = {
    "weather":     WeatherAgent(),
    "flight":      FlightAgent(),
    "attractions": AttractionsAgent(),
    "currency":    CurrencyAgent(),
    "timezone":    TimezoneAgent(),
}


# ── Node: pre_middleware ───────────────────────────────────────────────────────

def pre_middleware_node(state: TravelState) -> TravelState:
    """Clean raw STT input and inject conversation context.

    Writes: cleaned_input, context, metadata.
    Sub-agents never see raw user_input after this point.
    """
    result = _pre_mw.process(state["user_input"], state.get("conversation_history", []))
    return {**state, "cleaned_input": result["cleaned_input"]}


# ── Node: supervisor ──────────────────────────────────────────────────────────

def supervisor_node(state: TravelState) -> TravelState:
    """Detect all intents in cleaned_input and write them as a list.

    Writes: detected_intents (e.g. ["weather", "currency", "attractions"])
    """
    return _supervisor.route(state)


# ── Node: run_agents ──────────────────────────────────────────────────────────

def run_agents_node(state: TravelState) -> TravelState:
    """Run every detected sub-agent and collect their responses.

    Focused sub-queries for all intents are built in a single LLM call
    (batched) so N intents cost one round-trip instead of N round-trips.
    Today's date is passed so relative expressions like "next week" resolve
    to a concrete YYYY-MM-DD before reaching the flight agent.

    Writes: agent_responses ({intent: text}), tool_events ([{...}])
    """
    intents      = state.get("detected_intents", ["general"])
    query        = state["cleaned_input"]
    responses:   dict[str, str] = {}
    tool_events: list[dict]     = []
    model        = ModelLayer()

    # One LLM call to extract all focused sub-queries at once
    focused_map = _build_all_focused_queries(intents, query, model)

    for intent in intents:
        focused = focused_map.get(intent) or query
        label   = _INTENT_LABELS.get(intent, f"Processing {intent}")

        if intent == "general":
            try:
                t_gen = time.monotonic()
                response = model.invoke(focused)
                duration_ms = int((time.monotonic() - t_gen) * 1000)
                responses["general"] = response
                tool_events.append({
                    "tool_name":   "general_llm",
                    "label":       label,
                    "status":      "success",
                    "detail":      focused[:80],
                    "duration_ms": duration_ms,
                    "source":      "LLM Knowledge",
                })
                logger.info("run_agents_node | general | len=%d", len(response))
            except Exception as exc:
                logger.error("run_agents_node | general | failed: %s", exc)
                responses["general"] = "I couldn't process that. Please try again."
                tool_events.append({
                    "tool_name":   "general_llm",
                    "label":       label,
                    "status":      "error",
                    "detail":      str(exc),
                    "duration_ms": 0,
                    "source":      "LLM Knowledge",
                })
            continue

        agent = _AGENT_REGISTRY.get(intent)
        if not agent:
            logger.warning("run_agents_node | no agent for intent=%s", intent)
            continue

        try:
            t_agent = time.monotonic()
            response = agent.run(focused)
            duration_ms = int((time.monotonic() - t_agent) * 1000)
            responses[intent] = response
            tool_events.append({
                "tool_name":   f"get_{intent}",
                "label":       label,
                "status":      "success",
                "detail":      focused[:80],
                "duration_ms": duration_ms,
                "source":      _INTENT_SOURCES.get(intent, "External API"),
            })
            logger.info("run_agents_node | intent=%s | len=%d", intent, len(response))
        except Exception as exc:
            logger.error("run_agents_node | intent=%s | failed: %s", intent, exc)
            responses[intent] = f"Could not fetch {intent} information right now."
            tool_events.append({
                "tool_name":   f"get_{intent}",
                "label":       label,
                "status":      "error",
                "detail":      str(exc),
                "duration_ms": 0,
                "source":      _INTENT_SOURCES.get(intent, "External API"),
            })

    return {**state, "agent_responses": responses, "tool_events": tool_events}


# ── Node: merge_responses ─────────────────────────────────────────────────────

def merge_responses_node(state: TravelState) -> TravelState:
    """Combine all agent responses into one natural spoken response.

    Single-intent: returns the response directly — no merge needed.
    Multi-intent:  builds an ordered merge block (respecting the order intents
                   were detected) and asks the LLM to weave them together.
                   Word cap scales with the number of intents so every section
                   is guaranteed space in the final reply.

    Writes: final_response (pre-cleanup — post_middleware sanitises it next)
    """
    responses = state.get("agent_responses", {})

    if not responses:
        return {**state, "final_response": "I couldn't find information for your query. Please try again."}

    if len(responses) == 1:
        return {**state, "final_response": next(iter(responses.values()))}

    # Preserve the order intents were detected so the merge block matches
    # the order the user asked their questions.
    detected_order = state.get("detected_intents", list(responses.keys()))
    ordered_responses = {
        intent: responses[intent]
        for intent in detected_order
        if intent in responses
    }
    # Append any responses whose intent wasn't in detected_intents (safety net)
    for intent, resp in responses.items():
        if intent not in ordered_responses:
            ordered_responses[intent] = resp

    responses_text = "\n\n".join(
        f"=== {intent.upper()} ===\n{resp}"
        for intent, resp in ordered_responses.items()
    )

    # 80 words per intent gives each section breathing room without padding.
    word_limit = max(200, len(ordered_responses) * 80)

    prompt = _MERGE_TEMPLATE.format(
        user_input=state["cleaned_input"],
        responses_text=responses_text,
        word_limit=word_limit,
    )

    try:
        merged = ModelLayer().invoke(prompt)
        logger.info("merge_responses_node | intents=%s | word_limit=%d | merged len=%d",
                    list(ordered_responses.keys()), word_limit, len(merged))
        return {**state, "final_response": merged}
    except Exception as exc:
        logger.error("merge_responses_node | LLM merge failed: %s — joining directly", exc)
        joined = "  ".join(ordered_responses.values())
        return {**state, "final_response": joined}


# ── Node: post_middleware ─────────────────────────────────────────────────────

def post_middleware_node(state: TravelState) -> TravelState:
    """Strip markdown/URLs and truncate to TTS-safe length.

    Reads:  final_response
    Writes: final_response (cleaned in place)
    """
    cleaned = _post_mw.process(state.get("final_response", ""))
    logger.info("post_middleware_node | final_len=%d", len(cleaned))
    return {**state, "final_response": cleaned}


# ── Helper: batch focused sub-queries ────────────────────────────────────────

def _build_all_focused_queries(
    intents: list[str],
    full_query: str,
    model: ModelLayer,
) -> dict[str, str]:
    """Build focused sub-queries for ALL intents in a single LLM call.

    After the LLM produces each focused question, a Python validation pass
    corrects the two most common failure modes without a second LLM call:

    Flight:   LLM returns city names → look up IATA codes from _CITY_TO_IATA.
              LLM omits date / returns relative date → compute from today.
    Timezone: LLM returns "UTC" or a city name instead of an IANA string →
              look up correct IANA zone from _CITY_TO_TZ.

    Falls back to the full query for any intent where both LLM and validation fail.
    """
    today_str = _date.today().isoformat()
    intents_json = json.dumps(intents)
    prompt = (
        _BATCH_FOCUSED_TEMPLATE
        .replace("{today}", today_str)
        .replace("{full_query}", full_query)
        .replace("{intents_json}", intents_json)
    )

    llm_result: dict[str, str] = {}
    try:
        raw = model.invoke(prompt).strip()
        logger.info("_build_all_focused_queries | raw=%s", raw[:300])
        raw_clean = raw
        if "```" in raw_clean:
            raw_clean = raw_clean.split("```")[1]
            if raw_clean.startswith("json"):
                raw_clean = raw_clean[4:]
        llm_result = json.loads(raw_clean.strip())
    except Exception as exc:
        logger.warning("_build_all_focused_queries | LLM parse failed (%s) — using fallbacks", exc)

    result: dict[str, str] = {}
    for intent in intents:
        val = (llm_result.get(intent) or "").strip()

        if intent == "flight":
            val = _validate_flight_query(val, full_query, today_str)
        elif intent == "timezone":
            val = _validate_timezone_query(val, full_query)

        result[intent] = val if val else full_query
        logger.info("_build_all_focused_queries | intent=%s | focused=%s",
                    intent, result[intent][:80])
    return result


def _validate_flight_query(focused: str, full_query: str, today_str: str) -> str:
    """Ensure the flight focused query contains valid IATA codes and a date.

    Strategy:
      1. Find all 3-letter UPPERCASE words — treat them as candidate IATA codes.
      2. For any city name in full_query that maps in _CITY_TO_IATA, extract it.
      3. Ensure a YYYY-MM-DD date exists; if missing/relative → compute from today.
    """
    import re

    # Step 1 — collect IATA codes already in the focused string
    existing_iata = re.findall(r'\b([A-Z]{3})\b', focused)

    # Step 2 — find IATA codes by scanning the full query for known city names
    query_lower = full_query.lower()
    found_iata: list[str] = []
    # Sort by key length descending so "new york" matches before "york"
    for city in sorted(_CITY_TO_IATA, key=len, reverse=True):
        if city in query_lower and _CITY_TO_IATA[city] not in found_iata:
            found_iata.append(_CITY_TO_IATA[city])
        if len(found_iata) == 2:
            break

    # Merge: prefer IATA codes from the LLM; fill gaps from our map
    final_iata = existing_iata[:] if existing_iata else []
    for code in found_iata:
        if code not in final_iata:
            final_iata.append(code)
        if len(final_iata) == 2:
            break

    # Step 3 — ensure a concrete date
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', focused)
    if not date_match:
        # Compute relative dates from today
        query_lower2 = (focused + " " + full_query).lower()
        from datetime import date as _d, timedelta
        today = _d.fromisoformat(today_str)
        if "tomorrow" in query_lower2:
            flight_date = (today + timedelta(days=1)).isoformat()
        elif "next week" in query_lower2:
            flight_date = (today + timedelta(days=7)).isoformat()
        elif "next month" in query_lower2:
            flight_date = (today + timedelta(days=30)).isoformat()
        else:
            flight_date = (today + timedelta(days=7)).isoformat()  # default: next week
    else:
        flight_date = date_match.group()

    if len(final_iata) >= 2:
        return f"Find flights from {final_iata[0]} to {final_iata[1]} on {flight_date}."
    elif len(final_iata) == 1:
        # Only one airport resolved — keep whatever the LLM produced and append date
        if date_match:
            return focused
        return focused.rstrip(".") + f" on {flight_date}."
    else:
        # No IATA codes at all — return a best-effort string with the date
        if focused:
            return focused.rstrip(".") + f" on {flight_date}."
        return f"Find flights on {flight_date} based on: {full_query}"


def _validate_timezone_query(focused: str, full_query: str) -> str:
    """Ensure the timezone focused query mentions a resolvable city/timezone.

    If the LLM returned "UTC" or left it blank, scan the full query for a
    known city and substitute the correct IANA zone string.
    """
    # Extract city name from focused query — "What is the current local time in Bangkok?"
    import re
    city_match = re.search(r'(?:in|for)\s+([A-Za-z\s]+?)[\?\.]?$', focused, re.IGNORECASE)
    city_hint = city_match.group(1).strip().lower() if city_match else ""

    iana = _CITY_TO_TZ.get(city_hint, "")

    if not iana:
        # Scan full_query for any known city
        query_lower = full_query.lower()
        for city in sorted(_CITY_TO_TZ, key=len, reverse=True):
            if city in query_lower:
                iana = _CITY_TO_TZ[city]
                city_hint = city
                break

    if iana and city_hint:
        return f"What is the current local time in {city_hint.title()}? Use timezone {iana}."

    return focused if focused else full_query


# ── Node: summarize_response ──────────────────────────────────────────────────

def summarize_response_node(state: TravelState) -> TravelState:
    """Generate a short conversational summary from the optimised agent output.

    The optimised content (agent_responses joined, or final_response for
    single-intent) is passed back to the LLM with a summarization prompt.
    This produces the Summary View content — derived directly from the
    Optimised View so both tabs are always synchronized.

    Writes: summary_response (2-4 sentence plain-text, no markdown)
    """
    agent_responses = state.get("agent_responses", {})

    # Build the optimised content that the Summary should be derived from.
    # Multi-intent: join all agent responses; single-intent: use the one response.
    if len(agent_responses) > 1:
        optimised_text = "\n\n".join(
            f"=== {intent.upper()} ===\n{resp}"
            for intent, resp in agent_responses.items()
        )
    elif len(agent_responses) == 1:
        optimised_text = next(iter(agent_responses.values()))
    else:
        # Nothing to summarise — fall back to the merged final response
        optimised_text = state.get("final_response", "")

    if not optimised_text.strip():
        return {**state, "summary_response": state.get("final_response", "")}

    prompt = _SUMMARY_TEMPLATE.format(optimised_response=optimised_text)

    try:
        summary = ModelLayer().invoke(prompt).strip()
        # Strip any stray markdown that crept in despite the prompt instruction
        summary = summary.replace("**", "").replace("##", "").replace("* ", "").replace("- ", "")
        logger.info("summarize_response_node | summary_len=%d", len(summary))
        return {**state, "summary_response": summary}
    except Exception as exc:
        logger.error("summarize_response_node | LLM failed: %s — using final_response", exc)
        return {**state, "summary_response": state.get("final_response", "")}


# ── Graph assembly ─────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(TravelState)

    graph.add_node("pre_middleware",      pre_middleware_node)
    graph.add_node("supervisor",          supervisor_node)
    graph.add_node("run_agents",          run_agents_node)
    graph.add_node("merge_responses",     merge_responses_node)
    graph.add_node("post_middleware",     post_middleware_node)
    graph.add_node("summarize_response",  summarize_response_node)

    graph.add_edge(START,               "pre_middleware")
    graph.add_edge("pre_middleware",    "supervisor")
    graph.add_edge("supervisor",        "run_agents")
    graph.add_edge("run_agents",        "merge_responses")
    graph.add_edge("merge_responses",   "post_middleware")
    graph.add_edge("post_middleware",   "summarize_response")
    graph.add_edge("summarize_response", END)

    return graph.compile()


# Compiled once at import time — all requests share this instance.
_travel_graph = _build_graph()


# ── Public entry point ────────────────────────────────────────────────────────

def run_graph_full(user_input: str, history: list | None = None) -> dict:
    """Run the full multi-intent pipeline and return a structured result dict.

    Args:
        user_input: Raw transcribed text from the STT layer or /text/query body.
        history:    Conversation history [{role, content}, ...].

    Returns:
        {
            "response":    str,         # TTS-ready final response
            "intent":      str,         # Primary intent (first detected)
            "intents":     list[str],   # All detected intents
            "tool_events": list[dict],  # Tool activity for the frontend panel
        }
    """
    if history is None:
        history = []

    initial_state: TravelState = {
        "user_input":           user_input,
        "conversation_history": history,
        "cleaned_input":        "",
        "detected_intents":     [],
        "agent_responses":      {},
        "tool_events":          [],
        "final_response":       "",
        "summary_response":     "",
        "error":                "",
    }

    logger.info("run_graph_full | input=%s", user_input[:80])
    t0 = time.monotonic()
    result = _travel_graph.invoke(initial_state)
    elapsed = time.monotonic() - t0

    intents         = result.get("detected_intents", ["general"])
    final           = result.get("final_response", "Sorry, I couldn't process your request right now.")
    summary         = result.get("summary_response", final)
    tool_events     = result.get("tool_events", [])
    agent_responses = result.get("agent_responses", {})

    logger.info("run_graph_full | intents=%s | len=%d | elapsed=%.2fs", intents, len(final), elapsed)

    return {
        "response":         final,
        "summary_response": summary,
        "intent":           intents[0] if intents else "general",
        "intents":          intents,
        "tool_events":      tool_events,
        "agent_responses":  agent_responses,
    }
