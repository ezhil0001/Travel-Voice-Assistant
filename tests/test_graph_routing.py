# Tests the LangGraph supervisor + state machine in isolation.
#
# Intent detection is entirely LLM-based — keyword matching was removed because
# it only catches exact words, not meaning. All tests therefore mock the LLM
# response and verify the supervisor correctly parses and validates it.

import json
import logging
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)

from state.schema import TravelState
from agents.supervisor_agent import SupervisorAgent, VALID_INTENTS


# ── TravelState schema ─────────────────────────────────────────────────────────

def test_travel_state_has_required_keys():
    """TravelState TypedDict must define all fields the graph reads/writes."""
    required = {
        "user_input", "conversation_history", "cleaned_input",
        "detected_intents", "agent_responses", "tool_events",
        "final_response", "error",
    }
    annotations = TravelState.__annotations__
    for key in required:
        assert key in annotations, f"TravelState missing field: {key}"
    log.info("PASS | test_travel_state_has_required_keys")


# ── SupervisorAgent helpers ────────────────────────────────────────────────────

def _supervisor_returning(llm_output: str) -> SupervisorAgent:
    """Build a SupervisorAgent whose ModelLayer returns a fixed string."""
    sup = SupervisorAgent.__new__(SupervisorAgent)
    mock_model = MagicMock()
    mock_model.invoke.return_value = llm_output
    sup._model = mock_model
    return sup


def _make_state(cleaned_input: str) -> TravelState:
    return {
        "user_input": cleaned_input, "conversation_history": [],
        "cleaned_input": cleaned_input,
        "detected_intents": [], "agent_responses": {}, "tool_events": [],
        "final_response": "", "error": "",
    }


# ── LLM-based intent detection ────────────────────────────────────────────────

def test_supervisor_single_intent_weather():
    """LLM returning [\"weather\"] must produce detected_intents=[\"weather\"]."""
    sup = _supervisor_returning('["weather"]')
    result = sup.route(_make_state("What is the weather in Tokyo?"))
    assert result["detected_intents"] == ["weather"]
    log.info("PASS | test_supervisor_single_intent_weather")


def test_supervisor_single_intent_flight():
    sup = _supervisor_returning('["flight"]')
    result = sup.route(_make_state("Find flights from JFK to NRT"))
    assert result["detected_intents"] == ["flight"]
    log.info("PASS | test_supervisor_single_intent_flight")


def test_supervisor_single_intent_currency():
    sup = _supervisor_returning('["currency"]')
    result = sup.route(_make_state("How much is 500 USD in JPY?"))
    assert result["detected_intents"] == ["currency"]
    log.info("PASS | test_supervisor_single_intent_currency")


def test_supervisor_multi_intent():
    """LLM returning multiple intents must all appear in detected_intents."""
    sup = _supervisor_returning('["weather", "attractions", "currency"]')
    result = sup.route(_make_state(
        "Weather in Chennai, 5 days in Thailand, how much will it cost?"
    ))
    assert "weather"     in result["detected_intents"]
    assert "attractions" in result["detected_intents"]
    assert "currency"    in result["detected_intents"]
    log.info("PASS | test_supervisor_multi_intent")


def test_supervisor_llm_failure_returns_general():
    """If the LLM raises, supervisor must return ['general'] — not crash."""
    sup = SupervisorAgent.__new__(SupervisorAgent)
    mock_model = MagicMock()
    mock_model.invoke.side_effect = Exception("LLM offline")
    sup._model = mock_model
    result = sup.route(_make_state("some query"))
    assert result["detected_intents"] == ["general"]
    assert result["error"] == ""   # error field stays clean — LLM failure is handled internally
    log.info("PASS | test_supervisor_llm_failure_returns_general")


def test_supervisor_invalid_json_returns_general():
    """LLM returning garbage (not a JSON array) must produce ['general']."""
    sup = _supervisor_returning("sure, I think it might be weather related")
    result = sup.route(_make_state("What should I wear?"))
    assert result["detected_intents"] == ["general"]
    log.info("PASS | test_supervisor_invalid_json_returns_general")


def test_supervisor_unknown_intents_filtered_out():
    """Intent labels not in VALID_INTENTS must be silently dropped."""
    sup = _supervisor_returning('["weather", "unknown_domain", "currency"]')
    result = sup.route(_make_state("test"))
    assert "unknown_domain" not in result["detected_intents"]
    assert "weather"  in result["detected_intents"]
    assert "currency" in result["detected_intents"]
    log.info("PASS | test_supervisor_unknown_intents_filtered_out")


def test_supervisor_deduplicates_intents():
    """Duplicate labels in LLM output must be collapsed to one entry."""
    sup = _supervisor_returning('["weather", "weather", "currency"]')
    result = sup.route(_make_state("test"))
    assert result["detected_intents"].count("weather") == 1
    log.info("PASS | test_supervisor_deduplicates_intents")


def test_supervisor_uses_cleaned_input_not_user_input():
    """Supervisor must pass cleaned_input to the LLM, not raw user_input."""
    sup = SupervisorAgent.__new__(SupervisorAgent)
    mock_model = MagicMock()
    mock_model.invoke.return_value = '["currency"]'
    sup._model = mock_model

    state = _make_state("How much is 100 USD in Euros?")
    state["user_input"] = "UNCLEAN RAW INPUT"  # should be ignored
    sup.route(state)

    # The prompt passed to the model must contain cleaned_input, not user_input
    call_args = mock_model.invoke.call_args[0][0]
    assert "UNCLEAN RAW INPUT" not in call_args
    assert "100 USD" in call_args
    log.info("PASS | test_supervisor_uses_cleaned_input_not_user_input")


def test_valid_intents_set_is_complete():
    """VALID_INTENTS must cover all 6 routing domains."""
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
        "detected_intents": [], "agent_responses": {}, "tool_events": [],
        "final_response": "", "error": "",
    }
    result = pre_middleware_node(state)
    assert result["cleaned_input"] == "what is the weather in Tokyo"
    log.info("PASS | test_pre_middleware_node_cleans_input")


def test_post_middleware_node_strips_markdown():
    """post_middleware_node must strip markdown from final_response."""
    from graph.travel_graph import post_middleware_node
    state: TravelState = {
        "user_input": "", "conversation_history": [], "cleaned_input": "",
        "detected_intents": ["weather"], "agent_responses": {}, "tool_events": [],
        "final_response": "**Tokyo** is 18°C. Visit http://example.com for details.",
        "error": "",
    }
    result = post_middleware_node(state)
    assert "**" not in result["final_response"]
    assert "http" not in result["final_response"]
    log.info("PASS | test_post_middleware_node_strips_markdown")


def test_merge_responses_node_single_intent_returns_directly():
    """merge_responses_node must return the agent response directly for a single intent."""
    from graph.travel_graph import merge_responses_node
    state: TravelState = {
        "user_input": "Weather in Paris?", "conversation_history": [],
        "cleaned_input": "Weather in Paris?",
        "detected_intents": ["weather"],
        "agent_responses": {"weather": "Paris is 22°C and sunny today."},
        "tool_events": [], "final_response": "", "error": "",
    }
    result = merge_responses_node(state)
    assert result["final_response"] == "Paris is 22°C and sunny today."
    log.info("PASS | test_merge_responses_node_single_intent_returns_directly")


def test_merge_responses_node_empty_returns_fallback():
    """merge_responses_node must return a non-empty fallback when agent_responses is empty."""
    from graph.travel_graph import merge_responses_node
    state: TravelState = {
        "user_input": "", "conversation_history": [], "cleaned_input": "",
        "detected_intents": [], "agent_responses": {},
        "tool_events": [], "final_response": "", "error": "",
    }
    result = merge_responses_node(state)
    assert result["final_response"]
    log.info("PASS | test_merge_responses_node_empty_returns_fallback")


import json
import logging
