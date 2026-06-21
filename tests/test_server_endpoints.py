# Integration tests for server.py endpoints.
#
# All external dependencies (STT, TTS, LangGraph) are mocked so tests run
# without a live server, audio hardware, or real API keys. The ASGI test
# client from httpx drives the FastAPI app directly in-process.

import logging
import os
import sys
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_client():
    """Import server and return a TestClient — done lazily so mocks are in place first."""
    from server import app
    return TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────────

def test_health_endpoint_returns_ok():
    """/health must return status=ok with ollama and sarvam booleans."""
    with patch("server.http_requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        client = _get_client()
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "ollama" in body
    assert "sarvam" in body
    assert "timestamp" in body
    log.info("PASS | test_health_endpoint_returns_ok")


def test_health_ollama_down():
    """/health must report ollama=False when the Ollama endpoint is unreachable."""
    with patch("server.http_requests.get", side_effect=Exception("Connection refused")):
        client = _get_client()
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ollama"] is False
    log.info("PASS | test_health_ollama_down")


# ── /text/query ────────────────────────────────────────────────────────────────

def test_text_query_returns_response():
    """POST /text/query must pass text through run_graph and return JSON response."""
    with patch("server.run_graph", return_value="Tokyo is sunny today.") as mock_graph:
        client = _get_client()
        response = client.post(
            "/text/query",
            json={"text": "What is the weather in Tokyo?", "session_id": "test_session"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Tokyo is sunny today."
    assert body["session_id"] == "test_session"
    mock_graph.assert_called_once()
    log.info("PASS | test_text_query_returns_response")


def test_text_query_persists_history():
    """Consecutive /text/query calls must accumulate history for the same session."""
    sid = "history_test_session"
    responses = ["Tokyo is nice.", "It is 2PM in Tokyo."]

    with patch("server.run_graph", side_effect=responses):
        client = _get_client()
        # Clear any leftover session state
        from server import sessions
        sessions.pop(sid, None)

        client.post("/text/query", json={"text": "Tell me about Tokyo", "session_id": sid})
        client.post("/text/query", json={"text": "What time is it there?", "session_id": sid})

        history = sessions.get(sid, [])

    # 2 turns × 2 entries (user + assistant) = 4 entries
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    log.info("PASS | test_text_query_persists_history")


def test_text_query_history_capped_at_ten_turns():
    """Session history must never exceed MAX_HISTORY_TURNS × 2 entries."""
    sid = "cap_test_session"

    with patch("server.run_graph", return_value="ok"):
        client = _get_client()
        from server import sessions
        sessions.pop(sid, None)

        # Send 15 messages — only the last 10 turns should be kept
        for i in range(15):
            client.post("/text/query", json={"text": f"msg {i}", "session_id": sid})

        history = sessions.get(sid, [])

    assert len(history) <= 20  # 10 turns × 2 entries
    log.info("PASS | test_text_query_history_capped_at_ten_turns")


# ── /voice/query ───────────────────────────────────────────────────────────────

def test_voice_query_returns_audio():
    """POST /voice/query must return audio/wav bytes after full STT → graph → TTS pipeline."""
    fake_audio_out = b"RIFF....fakeaudioresponse"

    with patch("server.run_graph", return_value="Tokyo is sunny today."), \
         patch("server._stt") as mock_stt, \
         patch("server._tts") as mock_tts:

        mock_stt.transcribe.return_value = "What is the weather in Tokyo?"
        mock_tts.synthesize.return_value = fake_audio_out

        client = _get_client()
        response = client.post(
            "/voice/query",
            files={"audio_file": ("test.wav", b"fakeaudioinput", "audio/wav")},
            headers={"session-id": "voice_test"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content == fake_audio_out
    log.info("PASS | test_voice_query_returns_audio")


def test_voice_query_stt_failure_raises_422():
    """When STT returns empty string, /voice/query must return HTTP 422."""
    with patch("server._stt") as mock_stt:
        mock_stt.transcribe.return_value = ""
        client = _get_client()
        response = client.post(
            "/voice/query",
            files={"audio_file": ("test.wav", b"bad_audio", "audio/wav")},
        )

    assert response.status_code == 422
    log.info("PASS | test_voice_query_stt_failure_raises_422")


def test_voice_query_tts_failure_raises_502():
    """When TTS returns empty bytes, /voice/query must return HTTP 502."""
    with patch("server.run_graph", return_value="Some response"), \
         patch("server._stt") as mock_stt, \
         patch("server._tts") as mock_tts:

        mock_stt.transcribe.return_value = "What is the weather in Paris?"
        mock_tts.synthesize.return_value = b""

        client = _get_client()
        response = client.post(
            "/voice/query",
            files={"audio_file": ("test.wav", b"fakeaudio", "audio/wav")},
        )

    assert response.status_code == 502
    log.info("PASS | test_voice_query_tts_failure_raises_502")
