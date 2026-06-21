# Live integration tests — require the server to be running on localhost:8000.
#
# These are NOT part of the automated unit test suite. Run them manually after
# starting the server with: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
#
# They test the full end-to-end flow including real LLM calls, so they require
# valid API keys in .env and either Ollama running locally or Groq reachable.
#
# Usage:
#   uvicorn server:app --reload &
#   pytest tests/test_integration.py -v

import logging
import pytest
import requests

log = logging.getLogger(__name__)
BASE = "http://localhost:8000"


def _server_is_up() -> bool:
    """Check that OUR server is responding on /health with status=ok.
    A plain connection check is not enough — another process may be on port 8000.
    """
    try:
        r = requests.get(f"{BASE}/health", timeout=2)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


# Skip the entire module if the server isn't reachable — prevents CI failures
# when these tests are accidentally collected during automated runs.
pytestmark = pytest.mark.skipif(
    not _server_is_up(),
    reason="Live server not running on localhost:8000 — skipping integration tests",
)


def test_health():
    """/health must return status=ok."""
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "ollama" in data
    assert "sarvam" in data
    log.info("PASS | test_health | status=ok")


def test_weather_query():
    """/text/query must return a response for a weather question."""
    r = requests.post(
        f"{BASE}/text/query",
        json={"text": "What is the weather in Paris?", "session_id": "test1"},
        timeout=30,
    )
    assert r.status_code == 200
    assert "response" in r.json()
    assert len(r.json()["response"]) > 0
    log.info("PASS | test_weather_query | got response")


def test_currency_query():
    """/text/query must return a response for a currency conversion question."""
    r = requests.post(
        f"{BASE}/text/query",
        json={"text": "Convert 500 dollars to euros", "session_id": "test2"},
        timeout=30,
    )
    assert r.status_code == 200
    assert "response" in r.json()
    log.info("PASS | test_currency_query | got response")


def test_context_retention():
    """Second turn must reference context from the first turn in the same session."""
    sid = "test_context_retention"

    # Turn 1 — establish Tokyo as the destination
    requests.post(
        f"{BASE}/text/query",
        json={"text": "I want to visit Tokyo", "session_id": sid},
        timeout=30,
    )

    # Turn 2 — pronoun 'there' should resolve to Tokyo via conversation history
    r = requests.post(
        f"{BASE}/text/query",
        json={"text": "What time is it there?", "session_id": sid},
        timeout=30,
    )

    resp = r.json().get("response", "").lower()
    assert "tokyo" in resp or "japan" in resp or "jst" in resp, (
        f"Expected Tokyo context in response, got: {resp}"
    )
    log.info("PASS | test_context_retention | Tokyo context retained across turns")
