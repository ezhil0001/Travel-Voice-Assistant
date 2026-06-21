# LangGraph StateGraph that wires the full travel assistant pipeline.
#
# Node responsibilities (strict separation of concerns):
#
#   pre_middleware  — runs PreModelMiddleware via ServiceLayer stage 1.
#                    Writes cleaned_input, context, metadata into state.
#                    Sub-agents NEVER see raw user_input after this point.
#
#   supervisor      — reads cleaned_input only, writes detected_intent.
#                    Never calls tools; never sees conversation_history.
#
#   <intent>_node   — each reads only state["cleaned_input"].
#                    Runs agent.run() via ServiceLayer (middleware fully applied).
#                    Writes sub_agent_response into state.
#                    All retry/fallback logic stays inside BaseAgent._bound_chain.
#
#   post_middleware — reads sub_agent_response, writes final_response.
#                    Strips markdown/URLs, truncates to TTS-safe length.
#
# Edge topology:
#   START → pre_middleware → supervisor → conditional(detected_intent)
#   weather|flight|attractions|currency|timezone|general → post_middleware → END

import logging
import time
from langgraph.graph import StateGraph, START, END
from state.schema import TravelState
from agents.supervisor_agent import SupervisorAgent
from agents.weather_agent import WeatherAgent
from agents.flight_agent import FlightAgent
from agents.attractions_agent import AttractionsAgent
from agents.currency_agent import CurrencyAgent
from agents.timezone_agent import TimezoneAgent
from agents.model_layer import ModelLayer
from agents.service_layer import ServiceLayer
from middleware.pre_model import PreModelMiddleware
from middleware.post_model import PostModelMiddleware

logger = logging.getLogger(__name__)

# ── Singleton instances — built once, reused across all graph invocations ──────
_supervisor     = SupervisorAgent()
_weather_agent  = WeatherAgent()
_flight_agent   = FlightAgent()
_attractions_agent = AttractionsAgent()
_currency_agent = CurrencyAgent()
_timezone_agent = TimezoneAgent()
_model_layer    = ModelLayer()
_service_layer  = ServiceLayer()
_pre_mw         = PreModelMiddleware()
_post_mw        = PostModelMiddleware()


# ── Node definitions ───────────────────────────────────────────────────────────

def pre_middleware_node(state: TravelState) -> TravelState:
    """Clean raw STT input and inject conversation context.

    Runs PreModelMiddleware.before_model() directly — the graph owns stage 1
    here so the supervisor always receives cleaned_input, never raw user_input.
    Writes: cleaned_input, context, metadata.
    """
    updated = _pre_mw.before_model(dict(state))
    return {**state, **updated}


def supervisor_node(state: TravelState) -> TravelState:
    """Route based on cleaned_input → write detected_intent.

    The supervisor reads ONLY cleaned_input — it is structurally impossible
    for it to see raw conversation_history or call any tool from this node.
    """
    return _supervisor.route(state)


def _agent_node(state: TravelState, agent, agent_type: str) -> TravelState:
    """Generic agent execution node — shared by all 5 domain agents.

    Calls ServiceLayer.execute() which runs the full middleware pipeline
    (wrap_model_call + after_model) around agent.run(). The agent receives
    only cleaned_input — the raw state never flows into agent code.

    Writes: sub_agent_response.
    """
    try:
        response = _service_layer.execute(
            agent=agent,
            agent_type=agent_type,
            query=state["cleaned_input"],
            history=state.get("conversation_history", []),
        )
    except Exception as exc:
        logger.error("agent_node | type=%s | error=%s", agent_type, exc)
        response = f"Sorry, I couldn't complete your {agent_type} request right now."

    return {**state, "sub_agent_response": response, "error": ""}


def weather_node(state: TravelState) -> TravelState:
    return _agent_node(state, _weather_agent, "weather")


def flight_node(state: TravelState) -> TravelState:
    return _agent_node(state, _flight_agent, "flight")


def attractions_node(state: TravelState) -> TravelState:
    return _agent_node(state, _attractions_agent, "attractions")


def currency_node(state: TravelState) -> TravelState:
    return _agent_node(state, _currency_agent, "currency")


def timezone_node(state: TravelState) -> TravelState:
    return _agent_node(state, _timezone_agent, "timezone")


def general_node(state: TravelState) -> TravelState:
    """Handles conversational queries that don't require a specific tool.

    Uses ModelLayer directly (no tool binding needed) with a general
    travel assistant prompt assembled by DynamicPromptBuilder inside
    ServiceLayer's wrap_model_call hook.
    """
    try:
        response = _model_layer.invoke(state["cleaned_input"])
    except Exception as exc:
        logger.error("general_node | error=%s", exc)
        response = "I'm here to help with travel planning. What would you like to know?"

    return {**state, "sub_agent_response": response, "error": ""}


def post_middleware_node(state: TravelState) -> TravelState:
    """Sanitise the sub-agent response for TTS delivery.

    Runs PostModelMiddleware.after_model() — strips markdown/URLs and
    truncates to the configured character limit. Writes: final_response.
    """
    raw = state.get("sub_agent_response", "")
    updated = _post_mw.after_model({**state, "raw_response": raw})
    final = updated.get("cleaned_response", raw)
    return {**state, "final_response": final}


# ── Conditional routing edge ───────────────────────────────────────────────────

def _route_by_intent(state: TravelState) -> str:
    """Return the node name to route to based on detected_intent.

    This function is the ONLY place where detected_intent is read for routing.
    Adding a new domain = add one line here + one new node above.
    """
    intent = state.get("detected_intent", "general")
    routes = {
        "weather":     "weather_node",
        "flight":      "flight_node",
        "attractions": "attractions_node",
        "currency":    "currency_node",
        "timezone":    "timezone_node",
        "general":     "general_node",
    }
    return routes.get(intent, "general_node")


# ── Graph assembly ─────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(TravelState)

    # Register all nodes
    graph.add_node("pre_middleware",   pre_middleware_node)
    graph.add_node("supervisor",       supervisor_node)
    graph.add_node("weather_node",     weather_node)
    graph.add_node("flight_node",      flight_node)
    graph.add_node("attractions_node", attractions_node)
    graph.add_node("currency_node",    currency_node)
    graph.add_node("timezone_node",    timezone_node)
    graph.add_node("general_node",     general_node)
    graph.add_node("post_middleware",  post_middleware_node)

    # Linear edges: START → pre_middleware → supervisor
    graph.add_edge(START, "pre_middleware")
    graph.add_edge("pre_middleware", "supervisor")

    # Conditional edge: supervisor → one of the 6 domain nodes
    graph.add_conditional_edges(
        "supervisor",
        _route_by_intent,
        {
            "weather_node":     "weather_node",
            "flight_node":      "flight_node",
            "attractions_node": "attractions_node",
            "currency_node":    "currency_node",
            "timezone_node":    "timezone_node",
            "general_node":     "general_node",
        },
    )

    # All domain nodes converge at post_middleware → END
    for node in ("weather_node", "flight_node", "attractions_node",
                 "currency_node", "timezone_node", "general_node"):
        graph.add_edge(node, "post_middleware")

    graph.add_edge("post_middleware", END)

    return graph.compile()


# Compiled graph — imported by server.py and tests
travel_graph = _build_graph()


def run_graph(user_input: str, history: list | None = None) -> str:
    """Entry point for the full voice assistant pipeline.

    Args:
        user_input: Raw transcribed text from the STT layer.
        history:    Conversation history [{role, content}, ...].

    Returns:
        A clean, TTS-ready response string.
    """
    if history is None:
        history = []

    initial_state: TravelState = {
        "user_input":           user_input,
        "conversation_history": history,
        "cleaned_input":        "",
        "detected_intent":      "",
        "sub_agent_response":   "",
        "final_response":       "",
        "error":                "",
    }

    logger.info("run_graph | input=%s", user_input)
    t0 = time.monotonic()
    result = travel_graph.invoke(initial_state)
    elapsed = time.monotonic() - t0
    intent = result.get("detected_intent", "unknown")
    response_len = len(result.get("final_response", ""))
    logger.info(
        "run_graph | intent=%s | response_len=%d | elapsed=%.2fs",
        intent, response_len, elapsed,
    )
    return result.get("final_response", "Sorry, I couldn't process your request right now.")
