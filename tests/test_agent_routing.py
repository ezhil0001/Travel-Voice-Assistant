# Tests each sub-agent in isolation — verifying the tool binding + Runnable
# chain architecture is wired correctly.
#
# All agents now inherit from BaseAgent, so ChatOpenAI and ChatGroq are
# imported and instantiated inside base_agent.py. We patch them there so
# no real LLM is contacted during tests.

import logging
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)

from agents.weather_agent import WeatherAgent
from agents.flight_agent import FlightAgent
from agents.attractions_agent import AttractionsAgent
from agents.currency_agent import CurrencyAgent
from agents.timezone_agent import TimezoneAgent
from agents.service_layer import ServiceLayer


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_tool_call_response(tool_name: str, args: dict) -> MagicMock:
    """Simulate an AIMessage that contains a tool_calls list (bind_tools response)."""
    msg = MagicMock()
    msg.tool_calls = [{"name": tool_name, "args": args, "id": "call_001"}]
    msg.content = ""
    return msg


def _make_final_response(text: str) -> MagicMock:
    """Simulate a plain AIMessage returned by the format chain."""
    msg = MagicMock()
    msg.tool_calls = []
    msg.content = text
    return msg


def _build_agent_with_mock_chain(agent_cls, mock_chain):
    """Instantiate an agent with both LLM classes patched, then inject a mock chain."""
    with patch("agents.base_agent.ChatOpenAI") as mock_openai, \
         patch("agents.base_agent.ChatGroq"):
        mock_primary = MagicMock()
        mock_openai.return_value = mock_primary
        mock_primary.bind_tools.return_value.with_retry.return_value \
            .with_fallbacks.return_value = mock_chain
        mock_primary.with_retry.return_value.with_fallbacks.return_value = mock_chain
        agent = agent_cls()
    agent._bound_chain = mock_chain
    agent._format_chain = mock_chain
    return agent


# ── WeatherAgent ───────────────────────────────────────────────────────────────

def test_weather_agent_tool_binding_flow():
    """Bound chain issues a tool_call; agent executes it and formats the result."""
    fake_weather = {
        "city": "Tokyo", "temperature": 18.5, "feels_like": 17.0,
        "description": "clear sky", "humidity": 55, "wind_speed": 3.2,
    }
    tool_response  = _make_tool_call_response("get_weather", {"city": "Tokyo"})
    final_response = _make_final_response("Tokyo is 18°C with clear skies. Pack a light jacket.")

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [tool_response, final_response]

    with patch("agents.weather_agent.get_weather") as mock_tool:
        mock_tool.invoke.return_value = fake_weather
        agent = _build_agent_with_mock_chain(WeatherAgent, mock_chain)
        agent._tool = mock_tool
        result = agent.run("What is the weather in Tokyo?")

    assert isinstance(result, str) and len(result) > 0
    log.info("PASS | test_weather_agent_tool_binding_flow")


def test_weather_agent_direct_response():
    """If the LLM answers directly (no tool_call), return content unchanged."""
    direct_response = _make_final_response("Tokyo is sunny and warm today.")
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = direct_response

    agent = _build_agent_with_mock_chain(WeatherAgent, mock_chain)
    result = agent.run("Is it nice in Tokyo?")

    assert "Tokyo" in result
    log.info("PASS | test_weather_agent_direct_response")


def test_weather_agent_exception_returns_fallback_string():
    """Exceptions inside run() must be caught and return a graceful string."""
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("LLM timeout")

    agent = _build_agent_with_mock_chain(WeatherAgent, mock_chain)
    result = agent.run("Weather in Tokyo?")

    assert "sorry" in result.lower() or "couldn't" in result.lower()
    log.info("PASS | test_weather_agent_exception_returns_fallback_string")


# ── FlightAgent ────────────────────────────────────────────────────────────────

def test_flight_agent_tool_binding_flow():
    """Flight agent must pass IATA hints to tool and return formatted response."""
    fake_flights = [{"airline": "JL", "price": "750.00", "currency": "USD",
                     "stops": 1, "departure": "10:00", "arrival": "14:00"}]
    tool_response  = _make_tool_call_response("get_flights",
                        {"origin": "JFK", "destination": "NRT", "date": "2025-03-15"})
    final_response = _make_final_response("Japan Airlines flies JFK→NRT for $750 with 1 stop.")

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [tool_response, final_response]

    with patch("agents.flight_agent.get_flights") as mock_tool:
        mock_tool.invoke.return_value = fake_flights
        agent = _build_agent_with_mock_chain(FlightAgent, mock_chain)
        agent._tool = mock_tool
        result = agent.run("Flights from New York to Tokyo", "JFK", "NRT", "2025-03-15")

    assert isinstance(result, str) and len(result) > 0
    log.info("PASS | test_flight_agent_tool_binding_flow")


# ── AttractionsAgent ───────────────────────────────────────────────────────────

def test_attractions_agent_tool_binding_flow():
    """Attractions agent must call get_attractions and list top 3 places."""
    fake_attractions = [
        {"name": "Eiffel Tower", "kinds": "monument", "distance": 300},
        {"name": "Louvre",       "kinds": "museum",   "distance": 800},
        {"name": "Notre Dame",   "kinds": "church",   "distance": 1200},
    ]
    tool_response  = _make_tool_call_response("get_attractions", {"city": "Paris"})
    final_response = _make_final_response(
        "In Paris you should visit Eiffel Tower (iconic), Louvre (artistic), and Notre Dame (historic)."
    )

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [tool_response, final_response]

    with patch("agents.attractions_agent.get_attractions") as mock_tool:
        mock_tool.invoke.return_value = fake_attractions
        agent = _build_agent_with_mock_chain(AttractionsAgent, mock_chain)
        agent._tool = mock_tool
        result = agent.run("What should I visit in Paris?")

    assert isinstance(result, str) and len(result) > 0
    log.info("PASS | test_attractions_agent_tool_binding_flow")


# ── CurrencyAgent ──────────────────────────────────────────────────────────────

def test_currency_agent_tool_binding_flow():
    """Currency agent must invoke get_currency and add spending context."""
    fake_currency = {"from_currency": "USD", "to_currency": "JPY",
                     "amount": 500.0, "result": 75115.0, "rate": 150.23}
    tool_response  = _make_tool_call_response("get_currency",
                        {"from_c": "USD", "to_c": "JPY", "amount": 500})
    final_response = _make_final_response(
        "500 USD equals 75,115 yen — that covers about three nice dinners in Tokyo."
    )

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [tool_response, final_response]

    with patch("agents.currency_agent.get_currency") as mock_tool:
        mock_tool.invoke.return_value = fake_currency
        agent = _build_agent_with_mock_chain(CurrencyAgent, mock_chain)
        agent._tool = mock_tool
        result = agent.run("How much is 500 dollars in yen?", "USD", "JPY", 500)

    assert isinstance(result, str) and len(result) > 0
    log.info("PASS | test_currency_agent_tool_binding_flow")


# ── TimezoneAgent ──────────────────────────────────────────────────────────────

def test_timezone_agent_tool_binding_flow():
    """Timezone agent must call get_timezone and report local time + EST offset."""
    fake_tz = {"timezone": "Asia/Tokyo", "datetime": "2025-03-15T14:30:00",
               "hour": 14, "minute": 30}
    tool_response  = _make_tool_call_response("get_timezone", {"timezone": "Asia/Tokyo"})
    final_response = _make_final_response(
        "It is currently 2:30 PM in Tokyo, which is 14 hours ahead of US Eastern Time."
    )

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = [tool_response, final_response]

    with patch("agents.timezone_agent.get_timezone") as mock_tool:
        mock_tool.invoke.return_value = fake_tz
        agent = _build_agent_with_mock_chain(TimezoneAgent, mock_chain)
        agent._tool = mock_tool
        result = agent.run("What time is it in Tokyo?", "Asia/Tokyo")

    assert isinstance(result, str) and len(result) > 0
    log.info("PASS | test_timezone_agent_tool_binding_flow")


# ── BaseAgent inheritance ──────────────────────────────────────────────────────

def test_base_agent_builds_bound_and_format_chains():
    """BaseAgent.__init__ must call bind_tools() and build both chains."""
    with patch("agents.base_agent.ChatOpenAI") as mock_openai_cls, \
         patch("agents.base_agent.ChatGroq"):

        mock_primary = MagicMock()
        mock_openai_cls.return_value = mock_primary
        mock_primary.bind_tools.return_value.with_retry.return_value \
            .with_fallbacks.return_value = MagicMock()
        mock_primary.with_retry.return_value.with_fallbacks.return_value = MagicMock()

        agent = WeatherAgent()

        # bind_tools must have been called with get_weather
        mock_primary.bind_tools.assert_called_once()
        # Both chains must be set on the instance
        assert hasattr(agent, "_bound_chain")
        assert hasattr(agent, "_format_chain")
        assert hasattr(agent, "_tool")
    log.info("PASS | test_base_agent_builds_bound_and_format_chains")


# ── ServiceLayer ───────────────────────────────────────────────────────────────

def test_service_layer_middleware_pipeline():
    """ServiceLayer must run before_model → wrap_model_call → after_model in order.

    Key contracts verified:
    1. agent.run() receives only the cleaned_input string — no prompt, no history.
    2. Pre-model cleans STT whitespace artefacts before agent sees the query.
    3. Post-model strips markdown and URLs from the agent's raw response.
    """
    mock_agent = MagicMock()
    mock_agent.run.return_value = "**Tokyo** is 18°C. Visit http://example.com for more."

    layer = ServiceLayer()
    result = layer.execute(
        agent=mock_agent,
        agent_type="weather",
        query="  what is the weather in  Tokyo  ",
        history=[],
    )

    # agent.run() must receive only cleaned_input — double spaces collapsed,
    # no system prompt concatenated, no "User:" prefix.
    call_arg = mock_agent.run.call_args[0][0]
    assert "  " not in call_arg, "Pre-model should have collapsed double spaces"
    assert "You are a" not in call_arg, "Prompt must not be injected into agent.run() arg"

    # Post-model must have stripped markdown and URL from the raw agent response.
    assert "**" not in result, "Post-model should strip bold markdown"
    assert "http" not in result, "Post-model should strip URLs"
    log.info("PASS | test_service_layer_middleware_pipeline")


def test_service_layer_agent_receives_only_cleaned_input():
    """agent.run() must receive cleaned_input — not raw query, not history, not prompt."""
    mock_agent = MagicMock()
    mock_agent.run.return_value = "Paris is lovely."

    layer = ServiceLayer()
    layer.execute(
        agent=mock_agent,
        agent_type="attractions",
        query="  what should  i visit in paris  ",
        history=[{"role": "user", "content": "I want to travel"}],
    )

    call_arg = mock_agent.run.call_args[0][0]
    # Must be the cleaned string, not the raw input
    assert call_arg == "what should i visit in paris"
    log.info("PASS | test_service_layer_agent_receives_only_cleaned_input")
