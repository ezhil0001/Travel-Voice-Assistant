# Real-time text-to-speech streaming module.
#
# Exposes a FastAPI WebSocket endpoint that synthesises speech incrementally —
# the browser receives audio chunks as they are generated rather than waiting
# for the full response to be ready. This cuts perceived latency dramatically
# for long assistant replies.
#
# Architecture:
#
#   Browser  ←—— WebSocket (/tts/stream) ——→  FastAPI (this module)
#                                                     │
#                   Sarvam WSS  ←—————————————————————┤  primary
#                   Deepgram WSS ←————————————————————┘  fallback
#
# Event contract (browser → server):
#   JSON  { "type": "start_tts", "text": str, "language_code"?: str, "speaker"?: str }
#   JSON  { "type": "stop_tts" }
#
# Event contract (server → browser):
#   { "type": "tts_ready" }
#   { "type": "tts_audio",   "audio_base64": str, "format": str }
#   { "type": "tts_done" }
#   { "type": "tts_error",   "message": str }
#   { "type": "tts_stopped" }
#   { "type": "tts_disconnected", "code": int, "reason": str }
#
# Session lifecycle:
#   1. Browser connects to /tts/stream
#   2. Browser sends { type: "start_tts", text: "..." } → server opens upstream WS
#   3. Server sends config + text + flush to Sarvam/Deepgram
#   4. Upstream audio frames are base64-encoded and forwarded as tts_audio events
#   5. On synthesis completion the server sends tts_done and closes the upstream WS
#   6. Browser sends { type: "stop_tts" } to abort early (e.g. skip button)
#
# Provider audio formats:
#   Sarvam   — audio encoded as base64 inside JSON { type:"audio", data:{audio, content_type} }
#   Deepgram — raw binary WebSocket frames, base64-encoded here before forwarding
#
# Keepalive: upstream pings every PING_INTERVAL seconds — Sarvam idles out
# connections after ~30 s of inactivity during long graph processing times.

import asyncio
import base64
import json
import logging
import os
from typing import Optional
from urllib.parse import urlencode

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "test_results.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ── Tuning constants ───────────────────────────────────────────────────────────

PING_INTERVAL   = 20   # seconds between upstream keepalive pings
CONNECT_TIMEOUT = 10   # seconds to wait for upstream WS open


# ── Upstream URL builders ──────────────────────────────────────────────────────

def _sarvam_tts_url(model: str = "bulbul:v2") -> str:
    """Sarvam streaming TTS WebSocket URL.

    send_completion_event=true ensures we receive a final event frame so we
    know exactly when synthesis is done rather than relying on the WS closing.
    """
    params = urlencode({
        "model":                  model,
        "send_completion_event":  "true",
    })
    return f"wss://api.sarvam.ai/text-to-speech/ws?{params}"


def _deepgram_tts_url(model: str = "aura-asteria-en") -> str:
    """Deepgram streaming TTS WebSocket URL.

    linear16 at 22050 Hz matches what SarvamTTS produces in batch mode so
    the browser AudioContext doesn't need to handle two different formats.
    """
    params = urlencode({
        "model":       model,
        "encoding":    "linear16",
        "sample_rate": "22050",
    })
    return f"wss://api.deepgram.com/v1/speak?{params}"


# ── Upstream connectors ────────────────────────────────────────────────────────

async def _open_sarvam(language_code: str, speaker: str):
    """Connect to Sarvam TTS WebSocket and return the open socket.

    Raises on failure so the caller can fall back to Deepgram.
    """
    url = _sarvam_tts_url()
    ws = await asyncio.wait_for(
        websockets.connect(url, extra_headers={"Api-Subscription-Key": settings.SARVAM_API_KEY}),
        timeout=CONNECT_TIMEOUT,
    )
    # Sarvam requires a config frame immediately after the connection opens.
    # pitch/pace/loudness use Sarvam's documented defaults for bulbul:v2.
    config_frame = json.dumps({
        "type": "config",
        "data": {
            "model":               "bulbul:v2",
            "target_language_code": language_code,
            "speaker":             speaker,
            "speech_sample_rate":  22050,
            "enable_preprocessing": False,
            "pitch":    0,
            "pace":     1.0,
            "loudness": 1.0,
        },
    })
    await ws.send(config_frame)
    logger.info("tts_stream | upstream=Sarvam connected | lang=%s speaker=%s", language_code, speaker)
    return ws


async def _open_deepgram():
    """Connect to Deepgram TTS WebSocket and return the open socket.

    Raises on failure — caller treats this as a total provider failure.
    """
    url = _deepgram_tts_url()
    ws = await asyncio.wait_for(
        websockets.connect(url, extra_headers={"Authorization": f"Token {settings.DEEPGRAM_API_KEY}"}),
        timeout=CONNECT_TIMEOUT,
    )
    logger.info("tts_stream | upstream=Deepgram connected")
    return ws


async def _open_upstream(language_code: str, speaker: str):
    """Try Sarvam first, fall back to Deepgram.

    Returns (websocket, provider_name) or raises if both fail.
    """
    try:
        ws = await _open_sarvam(language_code, speaker)
        return ws, "sarvam"
    except Exception as exc:
        logger.warning("tts_stream | Sarvam upstream failed: %s — trying Deepgram", exc)

    ws = await _open_deepgram()
    return ws, "deepgram"


# ── Frame senders ──────────────────────────────────────────────────────────────

async def _send_text(ws, provider: str, text: str) -> None:
    """Send the text payload to the upstream provider in its expected format."""
    if provider == "sarvam":
        await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
        await ws.send(json.dumps({"type": "flush"}))
    else:  # deepgram
        # Clear any leftover buffered state from a previous synthesis in this session
        await ws.send(json.dumps({"type": "Clear"}))
        await ws.send(json.dumps({"type": "Speak", "text": text}))
        await ws.send(json.dumps({"type": "Flush"}))


# ── Frame normaliser ───────────────────────────────────────────────────────────

def _normalise_tts_frame(raw, provider: str) -> Optional[dict]:
    """Convert an upstream frame into our canonical event dict.

    Returns None for frames that should be silently ignored (metadata, ping, etc.).
    Returns a dict with type="tts_done" when synthesis is complete.
    """
    # ── Deepgram sends raw binary audio frames ─────────────────────────────────
    if isinstance(raw, (bytes, bytearray)):
        if provider == "deepgram" and len(raw) > 0:
            return {
                "type":         "tts_audio",
                "audio_base64": base64.b64encode(raw).decode("utf-8"),
                "format":       "linear16",
            }
        return None

    # ── Both providers send JSON text frames ───────────────────────────────────
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if provider == "sarvam":
        frame_type = data.get("type")

        if frame_type == "audio":
            audio_b64 = data.get("data", {}).get("audio", "")
            fmt       = data.get("data", {}).get("content_type", "wav")
            if audio_b64:
                return {
                    "type":         "tts_audio",
                    "audio_base64": audio_b64,
                    "format":       fmt,
                }
            return None

        if frame_type == "event":
            event_type = data.get("data", {}).get("event_type", "")
            if event_type == "final":
                return {"type": "tts_done"}
            return None  # intermediate events (e.g. 'started') — ignore

        if frame_type == "error":
            msg = data.get("data", {}).get("message", "Sarvam TTS error")
            return {"type": "tts_error", "message": msg}

        return None

    if provider == "deepgram":
        dg_type = data.get("type", "")

        if dg_type == "Flushed":
            # Deepgram sends Flushed after all audio for a Flush command is delivered
            return {"type": "tts_done"}

        if dg_type in ("Metadata", "Cleared", "Warning"):
            return None  # informational only

        return None

    return None


# ── WebSocket endpoint handler ─────────────────────────────────────────────────

async def tts_stream_endpoint(client: WebSocket):
    """FastAPI WebSocket handler — registered in server.py as /tts/stream.

    One handler per connected client. The upstream TTS connection is opened
    on start_tts and closed as soon as synthesis completes or stop_tts arrives,
    so no upstream socket stays open between user turns.

    The client can send multiple start_tts frames within the same WebSocket
    connection — useful for replaying TTS after an edit or retry.
    """
    await client.accept()
    logger.info("tts_stream | client connected")

    upstream: Optional[websockets.WebSocketClientProtocol] = None
    provider: str = ""
    ping_task: Optional[asyncio.Task] = None

    # ── inner helpers ──────────────────────────────────────────────────────────

    async def teardown(reason: str = "teardown"):
        nonlocal upstream, ping_task
        if ping_task and not ping_task.done():
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
        if upstream:
            try:
                await upstream.close(1000, reason)
            except Exception:
                pass
            upstream = None
        logger.info("tts_stream | upstream torn down (%s)", reason)

    async def heartbeat():
        while True:
            await asyncio.sleep(PING_INTERVAL)
            if upstream and upstream.open:
                try:
                    await upstream.ping()
                except Exception:
                    break

    async def forward_upstream_audio():
        """Read audio frames from the upstream provider and relay to the browser."""
        try:
            async for raw in upstream:
                event = _normalise_tts_frame(raw, provider)
                if event:
                    await client.send_text(json.dumps(event))
                    if event["type"] == "tts_done":
                        logger.info("tts_stream | synthesis complete")
                        break  # upstream will close normally after done
        except websockets.ConnectionClosed as exc:
            logger.info("tts_stream | upstream closed: code=%s reason=%s", exc.code, exc.reason)
            await client.send_text(json.dumps({
                "type":   "tts_disconnected",
                "code":   exc.code,
                "reason": exc.reason or "",
            }))
        except Exception as exc:
            logger.error("tts_stream | upstream relay error: %s", exc)
            await client.send_text(json.dumps({"type": "tts_error", "message": str(exc)}))

    # ── main receive loop ──────────────────────────────────────────────────────

    try:
        while True:
            message = await client.receive()

            if "text" not in message or not message["text"]:
                continue

            try:
                ctrl = json.loads(message["text"])
            except json.JSONDecodeError:
                continue

            ctrl_type = ctrl.get("type", "")

            # ── start_tts ─────────────────────────────────────────────────────
            if ctrl_type == "start_tts":
                text          = ctrl.get("text", "").strip()
                language_code = ctrl.get("language_code", "en-IN")
                speaker       = ctrl.get("speaker", "meera")

                if not text:
                    await client.send_text(json.dumps({
                        "type":    "tts_error",
                        "message": "start_tts requires a non-empty text field",
                    }))
                    continue

                # Tear down any in-flight session before starting a new one
                if upstream:
                    await teardown("restart")

                try:
                    upstream, provider = await _open_upstream(language_code, speaker)
                except Exception as exc:
                    logger.error("tts_stream | could not open upstream: %s", exc)
                    await client.send_text(json.dumps({
                        "type":    "tts_error",
                        "message": f"Could not connect to TTS provider: {exc}",
                    }))
                    continue

                await client.send_text(json.dumps({"type": "tts_ready"}))

                # Start keepalive ping and audio forwarding in parallel
                ping_task = asyncio.create_task(heartbeat())
                asyncio.create_task(forward_upstream_audio())

                # Send the text to the upstream provider
                try:
                    await _send_text(upstream, provider, text)
                except Exception as exc:
                    logger.error("tts_stream | text send failed: %s", exc)
                    await client.send_text(json.dumps({
                        "type":    "tts_error",
                        "message": f"Failed to send text to provider: {exc}",
                    }))
                    await teardown("text-send-error")

                logger.info(
                    "tts_stream | synthesis started | provider=%s len=%d",
                    provider, len(text),
                )

            # ── stop_tts ──────────────────────────────────────────────────────
            elif ctrl_type == "stop_tts":
                await teardown("client-stop")
                await client.send_text(json.dumps({"type": "tts_stopped"}))

    except WebSocketDisconnect:
        logger.info("tts_stream | client disconnected")
    except Exception as exc:
        logger.error("tts_stream | unexpected error: %s", exc)
    finally:
        await teardown("connection-end")
