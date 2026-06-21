# Tests the LangGraph supervisor + state machine in isolation.
#
# The graph wires real middleware, supervisor, and agent instances together.
# We mock only the LLM chains (ChatOpenAI/ChatGroq) and tool HTTP calls so
# the full routing, state mutation, and middleware transformation is exercised
# without hitting any real API endpoint.

import logging
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)

from state.schema import TravelState
from agents.supervisor_agent import SupervisorAgent, VALID_INTENTS


# ── TravelState schema ─────────────────────────────────────────────────────────

def test_travel_state_has_required_keys():
    """TravelState TypedDict must define all fields the graph reads/writes."""
    required = {
        "user_input", "conversation_history", "cleaned_input",
        "detected_intent", "sub_agent_response", "final_response", "error",
    }
    annotations = TravelState.__annotations__
    for key in required:
        assert key in annotations, f"TravelState missing field: {key}"
    log.info("PASS | test_travel_state_has_required_keys")


# ── SupervisorAgent ────────────────────────────────────────────────────────────

def _make_supervisor_with_intent(intent: str) -> SupervisorAgent:
    """Build a SupervisorAgent whose LLM chain returns a fixed intent string."""
    with patch("agents.supervisor_agent.ChatOpenAI") as mock_cls, \
         patch("agents.supervisor_agent.ChatGroq"):
        mock_primary = MagicMock()
        mock_cls.return_value = mock_primary
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(content=intent)
        mock_primary.with_retry.return_value.with_fallbacks.return_value = mock_chain
        sup = SupervisorAgent()
    sup._chain = MagicMock()
    sup._chain.invoke.return_value = MagicMock(content=intent)
    return sup


def test_supervisor_routes_weather():
    """Supervisor must write 'weather' to detected_intent for a weather query."""
    sup = _make_supervisor_with_intent("weather")
    state: TravelState = {
        "user_input": "What is the weather in Tokyo?",
        "conversation_history": [],
        "cleaned_input": "What is the weather in Tokyo?",
        "detected_intent": "", "sub_agent_response": "",
        "final_response": "", "error": "",
    }
    result = sup.route(state)
    assert result["detected_intent"] == "weather"
    log.info("PASS | test_supervisor_routes_weather")


def test_supervisor_routes_flight():
    sup = _make_supervisor_with_intent("flight")
    state: TravelState = {
        "user_input": "Find flights from JFK to NRT",
        "conversation_history": [], "cleaned_input": "Find flights from JFK to NRT",
        "detected_intent": "", "sub_agent_response": "", "final_response": "", "error": "",
    }
    result = sup.route(state)
    assert result["detected_intent"] == "flight"
    log.info("PASS | test_supervisor_routes_flight")


def test_supervisor_routes_currency():
    sup = _make_supervisor_with_intent("currency")
    state: TravelState = {
        "user_input": "How much is 500 USD in JPY?",
        "conversation_history": [], "cleaned_input": "How much is 500 USD in JPY?",
        "detected_intent": "", "sub_agent_response": "", "final_response": "", "error": "",
    }
    result = sup.route(state)
    assert result["detected_intent"] == "currency"
    log.info("PASS | test_supervisor_routes_currency")


def test_supervisor_invalid_intent_falls_back_to_general():
    """Unexpected LLM output must be silently corrected to 'general'."""
    sup = _make_supervisor_with_intent("nonsense_label")
    state: TravelState = {
        "user_input": "something weird", "conversation_history": [],
        "cleaned_input": "something weird",
        "detected_intent": "", "sub_agent_response": "", "final_response": "", "error": "",
    }
    result = sup.route(state)
    assert result["detected_intent"] == "general"
    log.info("PASS | test_supervisor_invalid_intent_falls_back_to_general")


def test_supervisor_never_reads_raw_user_input_when_cleaned_available():
    """Supervisor must read cleaned_input, not user_input, when both are present."""
    sup = _make_supervisor_with_intent("timezone")
    state: TravelState = {
        "user_input": "  UNCLEAN input with   spaces  ",
        "conversation_history": [],
        "cleaned_input": "What time is it in Tokyo?",
        "detected_intent": "", "sub_agent_response": "", "final_response": "", "error": "",
    }
    result = sup.route(state)
    # The chain was called — verify it was passed the cleaned version
    call_args = sup._chain.invoke.call_args[0][0]
    human_msg = call_args[1]
    assert "UNCLEAN" not in human_msg.content, "Supervisor must use cleaned_input, not user_input"
    assert result["detected_intent"] == "timezone"
    log.info("PASS | test_supervisor_never_reads_raw_user_input_when_cleaned_available")


def test_supervisor_exception_defaults_to_general():
    """If the LLM chain raises, supervisor must write 'general' and log the error."""
    with patch("agents.supervisor_agent.ChatOpenAI") as mock_cls, \
         patch("agents.supervisor_agent.ChatGroq"):
        mock_primary = MagicMock()
        mock_cls.return_value = mock_primary
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("LLM timeout")
        mock_primary.with_retry.return_value.with_fallbacks.return_value = mock_chain
        sup = SupervisorAgent()

    sup._chain = MagicMock()
    sup._chain.invoke.side_effect = Exception("LLM timeout")

    state: TravelState = {
        "user_input": "test", "conversation_history": [], "cleaned_input": "test",
        "detected_intent": "", "sub_agent_response": "", "final_response": "", "error": "",
    }
    result = sup.route(state)
    assert result["detected_intent"] == "general"
    assert result["error"] != ""
    log.info("PASS | test_supervisor_exception_defaults_to_general")


def test_valid_intents_set_is_complete():
    """VALID_INTENTS must cover all 6 routing domains the supervisor can classify."""
    expected = {"weather", "flight", "attractions", "currency", "timezone", "general"}
    assert expected == set(VALID_INTENTS)
    log.info("PASS | test_valid_intents_set_is_complete")


# ── Graph node isolation ───────────────────────────────────────────────────────

def test_pre_middleware_node_cleans_input():
    """pre_middleware_node must write cleaned_input into state."""
    from graph.travel_graph import pre_middleware_node
    state: TravelState = {
        "user_input": "  what is the weather in   Tokyo  ",
        "conversation_history": [], "cleaned_input": "",
        "detected_intent": "", "sub_agent_response": "", "final_response": "", "error": "",
    }
    result = pre_middleware_node(state)
    assert result["cleaned_input"] == "what is the weather in Tokyo"
    log.info("PASS | test_pre_middleware_node_cleans_input")


def test_post_middleware_node_strips_markdown():
    """post_middleware_node must strip markdown from sub_agent_response."""
    from graph.travel_graph import post_middleware_node
    state: TravelState = {
        "user_input": "", "conversation_history": [], "cleaned_input": "",
        "detected_intent": "weather",
        "sub_agent_response": "**Tokyo** is 18°C. Visit http://example.com for details.",
        "final_response": "", "error": "",
    }
    result = post_middleware_node(state)
    assert "**" not in result["final_response"]
    assert "http" not in result["final_response"]
    log.info("PASS | test_post_middleware_node_strips_markdown")


def test_route_by_intent_returns_correct_node():
    """_route_by_intent must return the correct node name for each intent."""
    from graph.travel_graph import _route_by_intent

    base: TravelState = {
        "user_input": "", "conversation_history": "", "cleaned_input": "",
        "sub_agent_response": "", "final_response": "", "error": "",
        "detected_intent": "",
    }

    cases = {
        "weather":     "weather_node",
        "flight":      "flight_node",
        "attractions": "attractions_node",
        "currency":    "currency_node",
        "timezone":    "timezone_node",
        "general":     "general_node",
        "unknown":     "general_node",   # safety fallback
    }
    for intent, expected_node in cases.items():
        state = {**base, "detected_intent": intent}
        assert _route_by_intent(state) == expected_node, f"Wrong node for intent={intent}"

    log.info("PASS | test_route_by_intent_returns_correct_node")
