# Tests for multi-intent detection and dynamic agent orchestration.
#
# These tests cover:
#   - Supervisor correctly identifying multiple intents in compound queries.
#   - run_agents_node calling every detected agent independently.
#   - merge_responses_node producing a merged reply for multi-intent results.
#   - run_graph_full returning all intents and tool_events in the response dict.
#
# All LLM and HTTP calls are mocked — nothing hits a real endpoint.

import json
import logging
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)


# ── Supervisor multi-intent detection ─────────────────────────────────────────

def _supervisor_llm_returning(json_array: str):
    """Build a SupervisorAgent whose LLM returns a pre-canned JSON array string."""
    from agents.supervisor_agent import SupervisorAgent
    sup = SupervisorAgent.__new__(SupervisorAgent)
    mock_model = MagicMock()
    mock_model.invoke.return_value = json_array
    sup._model = mock_model
    return sup


def test_supervisor_detects_weather_attractions_currency():
    """The exact compound query from the bug report must produce all three intents."""
    sup = _supervisor_llm_returning('["weather", "attractions", "currency"]')
    from state.schema import TravelState
    state: TravelState = {
        "user_input": (
            "Hi, I need exact weather report in Tamil Nadu, Chennai, "
            "and next day I am going to travel Thailand. "
            "So I have 5 days schedule and Thailand currency "
            "how will cost about 5 days expense. Then give."
        ),
        "conversation_history": [],
        "cleaned_input": (
            "Hi, I need exact weather report in Tamil Nadu, Chennai, "
            "and next day I am going to travel Thailand. "
            "So I have 5 days schedule and Thailand currency "
            "how will cost about 5 days expense. Then give."
        ),
        "detected_intents": [], "agent_responses": {}, "tool_events": [],
        "final_response": "", "summary_response": "", "error": "",
    }
    result = sup.route(state)
    intents = result["detected_intents"]
    assert "weather"     in intents, f"weather missing: {intents}"
    assert "attractions" in intents, f"attractions missing: {intents}"
    assert "currency"    in intents, f"currency missing: {intents}"
    log.info("PASS | test_supervisor_detects_weather_attractions_currency | intents=%s", intents)


def test_supervisor_single_intent_still_works():
    """A simple single-domain query must still return a single-element list."""
    sup = _supervisor_llm_returning('["weather"]')
    from state.schema import TravelState
    state: TravelState = {
        "user_input": "What is the weather in London?",
        "conversation_history": [],
        "cleaned_input": "What is the weather in London?",
        "detected_intents": [], "agent_responses": {}, "tool_events": [],
        "final_response": "", "summary_response": "", "error": "",
    }
    result = sup.route(state)
    assert result["detected_intents"] == ["weather"]
    log.info("PASS | test_supervisor_single_intent_still_works")


def test_supervisor_all_five_intents_detected():
    """A query touching all domains must produce all five intent labels."""
    sup = _supervisor_llm_returning(
        '["flight", "weather", "attractions", "timezone", "currency"]'
    )
    from state.schema import TravelState
    state: TravelState = {
        "user_input": (
            "Flights from Chennai to Tokyo, weather there, top places to visit, "
            "current local time, and USD to JPY rate?"
        ),
        "conversation_history": [],
        "cleaned_input": (
            "Flights from Chennai to Tokyo, weather there, top places to visit, "
            "current local time, and USD to JPY rate?"
        ),
        "detected_intents": [], "agent_responses": {}, "tool_events": [],
        "final_response": "", "summary_response": "", "error": "",
    }
    result = sup.route(state)
    intents = result["detected_intents"]
    assert len(intents) == 5, f"Expected 5 intents: {intents}"
    log.info("PASS | test_supervisor_all_five_intents_detected | intents=%s", intents)


def test_supervisor_deduplicates_intents():
    """Duplicate intent strings in LLM output must be de-duplicated."""
    sup = _supervisor_llm_returning('["weather", "weather", "currency"]')
    from state.schema import TravelState
    state: TravelState = {
        "user_input": "Weather and currency?", "conversation_history": [],
        "cleaned_input": "Weather and currency?",
        "detected_intents": [], "agent_responses": {}, "tool_events": [],
        "final_response": "", "summary_response": "", "error": "",
    }
    result = sup.route(state)
    assert result["detected_intents"].count("weather") == 1
    log.info("PASS | test_supervisor_deduplicates_intents")


# ── run_agents_node ────────────────────────────────────────────────────────────

def test_run_agents_node_calls_all_detected_agents():
    """run_agents_node must produce one agent_responses entry per detected intent."""
    from graph.travel_graph import run_agents_node
    from state.schema import TravelState

    state: TravelState = {
        "user_input": "Weather and currency", "conversation_history": [],
        "cleaned_input": "Weather in Chennai and currency for Thailand",
        "detected_intents": ["weather", "currency"],
        "agent_responses": {}, "tool_events": [], "final_response": "", "summary_response": "", "error": "",
    }

    batch_response = json.dumps({
        "weather":  "What is the current weather in Chennai?",
        "currency": "Convert 25000 INR to THB.",
    })

    with patch("graph.travel_graph.ModelLayer") as MockModel, \
         patch("graph.travel_graph._AGENT_REGISTRY", {
             "weather":  MagicMock(**{"run.return_value": "Chennai: 34°C"}),
             "currency": MagicMock(**{"run.return_value": "1 INR = 0.43 THB"}),
         }):
        MockModel.return_value.invoke.return_value = batch_response
        result = run_agents_node(state)

    assert "weather"  in result["agent_responses"]
    assert "currency" in result["agent_responses"]
    assert len(result["tool_events"]) == 2
    log.info("PASS | test_run_agents_node_calls_all_detected_agents")


def test_run_agents_node_records_tool_events():
    """Each successful agent run must append a tool_event with status=success."""
    from graph.travel_graph import run_agents_node
    from state.schema import TravelState

    state: TravelState = {
        "user_input": "", "conversation_history": [],
        "cleaned_input": "Weather in Tokyo?",
        "detected_intents": ["weather"],
        "agent_responses": {}, "tool_events": [], "final_response": "", "summary_response": "", "error": "",
    }

    batch_response = json.dumps({"weather": "What is the current weather in Tokyo?"})

    with patch("graph.travel_graph.ModelLayer") as MockModel, \
         patch("graph.travel_graph._AGENT_REGISTRY", {
             "weather": MagicMock(**{"run.return_value": "Tokyo: 18°C"}),
         }):
        MockModel.return_value.invoke.return_value = batch_response
        result = run_agents_node(state)

    assert result["tool_events"][0]["status"] == "success"
    assert result["tool_events"][0]["tool_name"] == "get_weather"
    log.info("PASS | test_run_agents_node_records_tool_events")


def test_run_agents_node_handles_agent_failure_gracefully():
    """If an agent raises, run_agents_node must record status=error and continue."""
    from graph.travel_graph import run_agents_node
    from state.schema import TravelState

    state: TravelState = {
        "user_input": "", "conversation_history": [],
        "cleaned_input": "weather and currency",
        "detected_intents": ["weather", "currency"],
        "agent_responses": {}, "tool_events": [], "final_response": "", "summary_response": "", "error": "",
    }

    batch_response = json.dumps({
        "weather":  "What is the weather?",
        "currency": "Convert 1000 INR to THB.",
    })

    with patch("graph.travel_graph.ModelLayer") as MockModel, \
         patch("graph.travel_graph._AGENT_REGISTRY", {
             "weather":  MagicMock(**{"run.side_effect": RuntimeError("API down")}),
             "currency": MagicMock(**{"run.return_value": "1 USD = 34 THB"}),
         }):
        MockModel.return_value.invoke.return_value = batch_response
        result = run_agents_node(state)

    events_by_tool = {e["tool_name"]: e for e in result["tool_events"]}
    assert events_by_tool["get_weather"]["status"]   == "error"
    assert events_by_tool["get_currency"]["status"]  == "success"
    log.info("PASS | test_run_agents_node_handles_agent_failure_gracefully")


# ── merge_responses_node ───────────────────────────────────────────────────────

def test_merge_responses_node_calls_llm_for_multiple_intents():
    """merge_responses_node must invoke ModelLayer when there are 2+ agent responses."""
    from graph.travel_graph import merge_responses_node
    from state.schema import TravelState

    state: TravelState = {
        "user_input": "Weather and currency?", "conversation_history": [],
        "cleaned_input": "Weather and currency?",
        "detected_intents": ["weather", "currency"],
        "agent_responses": {
            "weather":  "Chennai: 34°C humid",
            "currency": "1 INR = 0.43 THB",
        },
        "tool_events": [], "final_response": "", "summary_response": "", "error": "",
    }

    with patch("graph.travel_graph.ModelLayer") as MockModel:
        MockModel.return_value.invoke.return_value = "It is hot in Chennai and 1 INR is 0.43 THB."
        result = merge_responses_node(state)

    assert "Chennai" in result["final_response"] or "INR" in result["final_response"]
    MockModel.return_value.invoke.assert_called_once()
    log.info("PASS | test_merge_responses_node_calls_llm_for_multiple_intents")


# ── run_graph_full return shape ────────────────────────────────────────────────

def test_run_graph_full_returns_all_fields():
    """run_graph_full must return response, intent, intents, and tool_events keys."""
    from graph.travel_graph import run_graph_full

    # Mock the entire compiled graph so this test is fast and deterministic
    with patch("graph.travel_graph._travel_graph") as mock_graph:
        mock_graph.invoke.return_value = {
            "final_response":   "Chennai is hot, 1 INR is 0.43 THB.",
            "detected_intents": ["weather", "currency"],
            "tool_events": [
                {"tool_name": "get_weather",  "status": "success"},
                {"tool_name": "get_currency", "status": "success"},
            ],
            "agent_responses": {},
            "cleaned_input": "test",
            "error": "",
        }
        result = run_graph_full("test query")

    assert "response"    in result
    assert "intent"      in result
    assert "intents"     in result
    assert "tool_events" in result
    assert result["intent"]      == "weather"   # first detected intent
    assert result["intents"]     == ["weather", "currency"]
    assert len(result["tool_events"]) == 2
    log.info("PASS | test_run_graph_full_returns_all_fields")
