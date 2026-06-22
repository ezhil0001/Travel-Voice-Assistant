# Tests server.py endpoints using FastAPI's in-process test client.
#
# All external dependencies (STT, TTS, LangGraph) are mocked so these tests
# run without Ollama, audio hardware, or valid API keys. The goal is to verify
# routing, session persistence, error handling, and response shapes — not LLM behaviour.
#
# Voice endpoint notes:
#   /voice/query returns JSON (transcript + audio_base64), not raw audio/wav bytes.
#   Tests that previously asserted content-type == audio/wav have been updated to
#   match the current contract.

import base64
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
    """POST /text/query must route through run_graph_full and return response + intent."""
    with patch("server.run_graph_full", return_value={"response": "Tokyo is sunny today.", "intent": "weather"}):
        client = _get_client()
        response = client.post(
            "/text/query",
            json={"text": "What is the weather in Tokyo?", "session_id": "test_session"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Tokyo is sunny today."
    assert body["session_id"] == "test_session"
    assert body["intent"] == "weather"
    assert isinstance(body["tool_events"], list)
    log.info("PASS | test_text_query_returns_response")


def test_text_query_persists_history():
    """Consecutive /text/query calls for the same session must accumulate history."""
    sid = "history_test_session"
    turns = [
        {"response": "Tokyo is nice.", "intent": "general"},
        {"response": "It is 2PM in Tokyo.", "intent": "timezone"},
    ]

    with patch("server.run_graph_full", side_effect=turns):
        client = _get_client()
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
    """Session history must cap at MAX_HISTORY_TURNS regardless of how many messages are sent."""
    sid = "cap_test_session"

    with patch("server.run_graph_full", return_value={"response": "ok", "intent": "general"}):
        client = _get_client()
        from server import sessions
        sessions.pop(sid, None)

        for i in range(15):
            client.post("/text/query", json={"text": f"msg {i}", "session_id": sid})

        history = sessions.get(sid, [])

    assert len(history) <= 20  # 10 turns × 2 entries (user + assistant)
    log.info("PASS | test_text_query_history_capped_at_ten_turns")


# ── /voice/query ───────────────────────────────────────────────────────────────

def test_voice_query_returns_json_with_transcript_and_audio():
    """/voice/query must return JSON containing transcript, response, intent, and audio_base64.

    The endpoint was changed from StreamingResponse (raw audio bytes) to JSON
    so the client can read the transcript and intent alongside playing audio —
    all from a single request. Tests that previously checked content-type audio/wav
    were updated to match the current contract.
    """
    fake_audio_out = b"RIFF....fakeaudioresponse"

    with patch("server.run_graph_full", return_value={"response": "Tokyo is sunny today.", "intent": "weather"}), \
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
    body = response.json()
    assert body["transcript"] == "What is the weather in Tokyo?"
    assert body["response"] == "Tokyo is sunny today."
    assert body["intent"] == "weather"
    # audio_base64 must decode back to the original bytes
    assert base64.b64decode(body["audio_base64"]) == fake_audio_out
    log.info("PASS | test_voice_query_returns_json_with_transcript_and_audio")


def test_voice_query_stt_failure_raises_422():
    """When STT returns an empty transcript, /voice/query must respond 422.

    An empty transcript means the audio was either silent or too short for
    the STT engine to process. A 422 tells the client to prompt the user
    to try speaking again rather than silently swallowing the error.
    """
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
    """When TTS returns empty bytes, /voice/query must respond 502.

    Empty TTS output indicates a provider-side failure. Returning 502 lets
    the client distinguish between a bad request (4xx) and an upstream
    dependency failure that may resolve on retry.
    """
    with patch("server.run_graph_full", return_value={"response": "Some response", "intent": "general"}), \
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
