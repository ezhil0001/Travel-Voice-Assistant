# FastAPI entry point — exposes the travel assistant over HTTP.
#
# Three endpoints:
#   POST /voice/query  — full voice pipeline: STT → graph → TTS → audio bytes
#   POST /text/query   — text-only: skips STT/TTS, useful for testing and Retell webhooks
#   GET  /health       — liveness check for Ollama and Sarvam connectivity
#
# Session state is kept in a process-local dict keyed by session_id.
# History is capped at 10 turns so prompts never exceed the model's context window.

import logging
import os
import time
from datetime import datetime
from typing import Optional

import requests as http_requests
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from graph.travel_graph import run_graph
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
    detected_intent: Optional[str] = None


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
    """Full voice pipeline: audio → STT → graph → TTS → audio bytes.

    Accepts a multipart form upload containing raw audio (WAV/MP3).
    Returns an audio/wav StreamingResponse the client can play directly.
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

    # Step 3 — Run the LangGraph pipeline
    try:
        response_text = run_graph(user_text, history)
    except Exception as exc:
        logger.error("%s | run_graph failed: %s", ts, exc)
        raise HTTPException(status_code=500, detail="Processing failed. Please try again.")
    logger.info("%s | Graph response: %s", ts, response_text)

    # Step 4 — Persist history (capped at MAX_HISTORY_TURNS)
    _update_history(session_id, user_text, response_text)

    # Step 5 — Synthesise speech
    audio_out = _tts.synthesize(response_text)
    logger.info("%s | TTS synthesised %d bytes", ts, len(audio_out))

    if not audio_out:
        raise HTTPException(status_code=502, detail="TTS synthesis failed. Please try again.")

    elapsed = time.monotonic() - t0
    logger.info("%s | /voice/query | session=%s | elapsed=%.2fs", ts, session_id, elapsed)

    # Step 6 — Stream audio back to client
    return StreamingResponse(
        iter([audio_out]),
        media_type="audio/wav",
        headers={"X-Session-Id": session_id},
    )


@app.post("/text/query", response_model=TextQueryResponse)
async def text_query(body: TextQueryRequest):
    """Text-only pipeline — skips STT and TTS.

    Useful for:
      - Integration testing without audio hardware
      - Retell AI webhook callbacks (Retell sends/receives text)
      - Dashboard testing during development
    """
    ts = datetime.utcnow().isoformat()
    t0 = time.monotonic()
    logger.info("%s | /text/query | session=%s | text=%s", ts, body.session_id, body.text)

    history = _get_history(body.session_id)
    try:
        response_text = run_graph(body.text, history)
    except Exception as exc:
        logger.error("%s | /text/query run_graph failed: %s", ts, exc)
        raise HTTPException(status_code=500, detail="Processing failed. Please try again.")
    _update_history(body.session_id, body.text, response_text)

    elapsed = time.monotonic() - t0
    logger.info("%s | /text/query | session=%s | elapsed=%.2fs | response=%s",
                ts, body.session_id, elapsed, response_text)
    return TextQueryResponse(response=response_text, session_id=body.session_id)


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
