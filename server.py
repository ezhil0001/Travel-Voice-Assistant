# FastAPI entry point — exposes the travel assistant over HTTP.
#
# Three endpoints:
#   POST /voice/query  — full voice pipeline: STT → graph → TTS → base64 JSON
#                        (changed from raw StreamingResponse so the Angular client
#                         can receive transcript, intent, and audio in one response)
#   POST /text/query   — text-only: skips STT/TTS, useful for testing and Retell webhooks
#   GET  /health       — liveness check for Ollama and Sarvam connectivity
#
# Session state is kept in a process-local dict keyed by session_id.
# History is capped at 10 turns so prompts never exceed the model's context window.

import base64
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

import requests as http_requests
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from graph.travel_graph import run_graph, run_graph_full
from voice.stt import STTProvider
from voice.tts import TTSProvider

# ── Logging ────────────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/test_results.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App and session store ──────────────────────────────────────────────────────

app = FastAPI(
    title="Travel Voice Assistant",
    description="LangGraph + LangChain multi-agent travel assistant with voice I/O",
    version="1.0.0",
)

# Allow the Angular dev server (port 4200) and any production origin to call
# the API from the browser. Without this every fetch/XHR from the frontend is
# blocked before it even reaches the endpoint handlers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store — maps session_id → conversation history list.
# Each entry is {role: "user"|"assistant", content: str}.
# Capped at MAX_HISTORY_TURNS to prevent unbounded prompt growth.
sessions: dict[str, list] = {}
MAX_HISTORY_TURNS = 10

_stt = STTProvider()
_tts = TTSProvider()


# ── Request/response schemas ───────────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    text: str
    session_id: str = "default"


# tool_events is an empty list for now — the graph doesn't yet surface per-tool
# call metadata to the HTTP layer. The field is included so the Angular service
# contract is satisfied without any client-side workarounds.
class TextQueryResponse(BaseModel):
    response: str
    session_id: str
    intent: str = "general"          # renamed from detected_intent to match Angular contract
    tool_events: List[dict] = []     # reserved for future tool-call telemetry


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_history(session_id: str) -> list:
    return sessions.get(session_id, [])


def _update_history(session_id: str, user_text: str, assistant_text: str) -> None:
    history = sessions.setdefault(session_id, [])
    history.append({"role": "user",      "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    # Keep only the most recent MAX_HISTORY_TURNS turns (each turn = 2 entries)
    if len(history) > MAX_HISTORY_TURNS * 2:
        sessions[session_id] = history[-(MAX_HISTORY_TURNS * 2):]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/voice/query")
async def voice_query(
    audio_file: UploadFile = File(...),
    session_id: str = Header(default="default"),
):
    """Full voice pipeline: audio → STT → graph → TTS → JSON.

    Accepts a multipart form upload containing raw audio (WAV/MP3).

    Previously this returned a raw audio/wav StreamingResponse, which meant
    the client had no way to access the transcript, response text, or intent
    alongside the audio. Changed to return a JSON body so the Angular client
    can populate message bubbles, intent badges, and tool activity panels
    without a separate text/query call.

    Response shape:
        {
            "transcript":   str   — what the STT heard,
            "response":     str   — the assistant's text reply,
            "intent":       str   — weather/flight/attractions/currency/timezone/general,
            "tool_events":  list  — reserved for future tool-call telemetry,
            "audio_base64": str   — WAV audio bytes base64-encoded,
            "session_id":   str,
        }
    """
    ts = datetime.utcnow().isoformat()
    t0 = time.monotonic()
    logger.info("%s | /voice/query | session=%s", ts, session_id)

    # Step 1 — Transcribe audio to text
    audio_bytes = await audio_file.read()
    user_text = _stt.transcribe(audio_bytes)
    logger.info("%s | STT result: %s", ts, user_text)

    if not user_text:
        raise HTTPException(status_code=422, detail="Could not transcribe audio. Please try again.")

    # Step 2 — Load session history
    history = _get_history(session_id)

    # Step 3 — Run the LangGraph pipeline (returns response + intent together)
    try:
        result = run_graph_full(user_text, history)
    except Exception as exc:
        logger.error("%s | run_graph_full failed: %s", ts, exc)
        raise HTTPException(status_code=500, detail="Processing failed. Please try again.")

    response_text = result["response"]
    intent        = result["intent"]
    logger.info("%s | Graph response: %s | intent: %s", ts, response_text, intent)

    # Step 4 — Persist history (capped at MAX_HISTORY_TURNS)
    _update_history(session_id, user_text, response_text)

    # Step 5 — Synthesise speech
    audio_out = _tts.synthesize(response_text)
    logger.info("%s | TTS synthesised %d bytes", ts, len(audio_out))

    if not audio_out:
        raise HTTPException(status_code=502, detail="TTS synthesis failed. Please try again.")

    elapsed = time.monotonic() - t0
    logger.info("%s | /voice/query | session=%s | elapsed=%.2fs", ts, session_id, elapsed)

    # Step 6 — Return JSON so the Angular client gets transcript + audio in one shot.
    # The audio is base64-encoded because JSON cannot carry raw binary data.
    # The client decodes it with atob() and creates an object URL for HTMLAudioElement.
    return {
        "transcript":   user_text,
        "response":     response_text,
        "intent":       intent,
        "tool_events":  [],
        "audio_base64": base64.b64encode(audio_out).decode("utf-8"),
        "session_id":   session_id,
    }


@app.post("/text/query", response_model=TextQueryResponse)
async def text_query(body: TextQueryRequest):
    """Text-only pipeline — skips STT and TTS.

    Useful for:
      - Integration testing without audio hardware
      - Retell AI webhook callbacks (Retell sends/receives text)
      - Dashboard testing during development

    Returns JSON with: response, session_id, intent, tool_events.
    The intent field lets the Angular client render the correct badge on the
    message bubble without a second round-trip.
    """
    ts = datetime.utcnow().isoformat()
    t0 = time.monotonic()
    logger.info("%s | /text/query | session=%s | text=%s", ts, body.session_id, body.text)

    history = _get_history(body.session_id)
    try:
        result = run_graph_full(body.text, history)
    except Exception as exc:
        logger.error("%s | /text/query run_graph_full failed: %s", ts, exc)
        raise HTTPException(status_code=500, detail="Processing failed. Please try again.")

    response_text = result["response"]
    intent        = result["intent"]

    _update_history(body.session_id, body.text, response_text)

    elapsed = time.monotonic() - t0
    logger.info(
        "%s | /text/query | session=%s | intent=%s | elapsed=%.2fs | response=%s",
        ts, body.session_id, intent, elapsed, response_text,
    )
    return TextQueryResponse(
        response=response_text,
        session_id=body.session_id,
        intent=intent,
        tool_events=[],
    )


@app.get("/health")
async def health():
    """Liveness check — reports connectivity to Ollama and Sarvam.

    Ollama is checked by hitting its /api/tags endpoint (returns model list).
    Sarvam is checked by verifying the API key is configured — we don't make
    a live call here to avoid billable usage on every health ping.
    """
    ollama_ok = False
    try:
        r = http_requests.get(
            f"{settings.OLLAMA_BASE_URL}/api/tags",
            timeout=3,
        )
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False

    sarvam_ok = bool(settings.SARVAM_API_KEY and settings.SARVAM_API_KEY != "your_key")

    logger.info("health | ollama=%s sarvam=%s", ollama_ok, sarvam_ok)

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "ollama": ollama_ok,
        "sarvam": sarvam_ok,
    }
