# FastAPI entry point — exposes the travel assistant over HTTP and WebSocket.
#
# Endpoints:
#   POST /voice/query  — full voice pipeline: STT → graph → TTS → base64 JSON
#   POST /text/query   — text-only pipeline: skips STT/TTS, returns JSON
#   WS   /voice/stream — real-time streaming STT (PCM chunks → transcript events)
#   WS   /tts/stream   — real-time streaming TTS (text → audio chunk events)
#   GET  /health       — liveness check for Ollama and Sarvam connectivity
#
# Session state is kept in a process-local dict keyed by session_id.
# History is capped at 10 turns so prompts never exceed the model's context window.

import base64
import logging
import os
import time
from datetime import datetime
from typing import List

import requests as http_requests
from fastapi import FastAPI, File, Header, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from graph.travel_graph import run_graph_full
from voice.stt import STTProvider
from voice.stt_stream import stt_stream_endpoint
from voice.tts import TTSProvider
from voice.tts_stream import tts_stream_endpoint

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


class TextQueryResponse(BaseModel):
    response: str
    session_id: str
    intent: str = "general"
    intents: List[str] = []
    tool_events: List[dict] = []
    agent_responses: dict = {}
    summary_response: str = ""


class TtsSynthesizeRequest(BaseModel):
    text: str


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
    """Full voice pipeline: audio → STT → LangGraph → TTS → JSON.

    Accepts a multipart/form-data upload containing a WAV or MP3 recording.
    Returns a single JSON response so the client can populate the message bubble,
    intent badge, and audio player from one round-trip — no follow-up request needed.

    Response shape:
        transcript:   what the STT heard
        response:     the assistant's text reply
        intent:       weather / flight / attractions / currency / timezone / general
        tool_events:  list of tool calls made during processing (empty until graph exposes them)
        audio_base64: WAV bytes base64-encoded — decode with atob() and play via AudioContext
        session_id:   echoed back so the client can correlate multi-turn history
    """
    ts = datetime.utcnow().isoformat()
    t0 = time.monotonic()
    logger.info("%s | /voice/query | session=%s", ts, session_id)

    audio_bytes = await audio_file.read()
    user_text = _stt.transcribe(audio_bytes)
    logger.info("%s | STT result: %s", ts, user_text)

    if not user_text:
        raise HTTPException(status_code=422, detail="Could not transcribe audio. Please try again.")

    history = _get_history(session_id)

    try:
        result = run_graph_full(user_text, history)
    except Exception as exc:
        logger.error("%s | run_graph_full failed: %s", ts, exc)
        raise HTTPException(status_code=500, detail="Processing failed. Please try again.")

    response_text = result["response"]
    intent        = result["intent"]
    logger.info("%s | Graph response: %s | intent: %s", ts, response_text, intent)

    _update_history(session_id, user_text, response_text)

    audio_out = _tts.synthesize(response_text)
    logger.info("%s | TTS synthesised %d bytes", ts, len(audio_out))

    if not audio_out:
        raise HTTPException(status_code=502, detail="TTS synthesis failed. Please try again.")

    elapsed = time.monotonic() - t0
    logger.info("%s | /voice/query | session=%s | elapsed=%.2fs", ts, session_id, elapsed)

    return {
        "transcript":        user_text,
        "response":          response_text,
        "intent":            intent,
        "intents":           result.get("intents", [intent]),
        "tool_events":       result.get("tool_events", []),
        "agent_responses":   result.get("agent_responses", {}),
        "summary_response":  result.get("summary_response", response_text),
        "audio_base64":      base64.b64encode(audio_out).decode("utf-8"),
        "session_id":        session_id,
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
        intents=result.get("intents", [intent]),
        tool_events=result.get("tool_events", []),
        agent_responses=result.get("agent_responses", {}),
        summary_response=result.get("summary_response", response_text),
    )


@app.post("/tts/synthesize")
async def tts_synthesize(body: TtsSynthesizeRequest):
    """Synthesise arbitrary text to speech without running the LangGraph pipeline.

    Used by the frontend to auto-play the welcome message on page load so the
    user hears the greeting without pressing the mic button.

    Returns:
        audio_base64: WAV bytes base64-encoded — decode with atob() and play
                      via HTMLAudioElement or AudioContext.
    """
    ts = datetime.utcnow().isoformat()
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")

    audio_out = _tts.synthesize(text)
    if not audio_out:
        raise HTTPException(status_code=502, detail="TTS synthesis failed.")

    logger.info("%s | /tts/synthesize | %d bytes", ts, len(audio_out))
    return {"audio_base64": base64.b64encode(audio_out).decode("utf-8")}


@app.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket):
    """Real-time streaming STT endpoint.

    Proxies binary audio chunks from the browser to Sarvam (or Deepgram as
    fallback) and forwards transcript frames back as JSON events.

    Event contract — see voice/stt_stream.py for full documentation.
    """
    await stt_stream_endpoint(websocket)


@app.websocket("/tts/stream")
async def tts_stream(websocket: WebSocket):
    """Real-time streaming TTS endpoint.

    Accepts a start_tts JSON frame with the assistant's response text,
    opens an upstream WebSocket to Sarvam (or Deepgram as fallback),
    and streams base64-encoded audio chunks back to the browser as they
    are generated. The browser can play each chunk immediately rather than
    waiting for the full audio file to be synthesised.

    Event contract — see voice/tts_stream.py for full documentation.
    """
    await tts_stream_endpoint(websocket)


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
