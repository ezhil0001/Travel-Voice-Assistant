# Real-time speech-to-text streaming module.
#
# Exposes a FastAPI WebSocket endpoint that acts as a transparent proxy between
# the browser and the upstream Sarvam (primary) or Deepgram (fallback) streaming
# STT service. The browser sends raw PCM chunks; transcription results flow back
# as JSON events without any polling round-trips.
#
# Architecture:
#
#   Browser  ←—— WebSocket (/voice/stream) ——→  FastAPI (this module)
#                                                      │
#                    Sarvam WSS  ←——————————————————————┤  primary
#                    Deepgram WSS ←——————————————————————┘  fallback
#
# Event contract (browser → server):
#   JSON  { "type": "start_stt",  "session_id": str, "language_code": str }
#   bytes  <raw PCM audio chunk>
#   JSON  { "type": "stop_stt" }
#
# Event contract (server → browser):
#   { "type": "stt_ready" }
#   { "type": "stt_interim",  "transcript": str, "is_final": false }
#   { "type": "stt_final",    "transcript": str, "is_final": true  }
#   { "type": "stt_error",    "message": str }
#   { "type": "stt_stopped" }
#   { "type": "stt_disconnected", "code": int, "reason": str }
#
# Session lifecycle:
#   1. Browser connects to /voice/stream
#   2. Browser sends { type: "start_stt", ... } → opens upstream WS to Sarvam/Deepgram
#   3. Browser streams raw PCM chunks as binary WebSocket frames
#   4. Upstream transcript frames are normalised and forwarded as stt_interim/stt_final events
#   5. Browser (or silence timeout) sends { type: "stop_stt" } → upstream connection torn down
#
# Back-pressure: if the upstream socket's write buffer exceeds BUFFER_LIMIT bytes,
# the chunk is dropped rather than queued — prevents unbounded memory growth when
# Sarvam is slower than the microphone capture rate.
#
# Keepalive: a ping is sent upstream every PING_INTERVAL seconds so the connection
# survives load-balancer idle timeouts without the browser having to reconnect.

import asyncio
import base64
import json
import logging
import os
import ssl
from typing import Optional

import certifi
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

PING_INTERVAL  = 20        # seconds between upstream keepalive pings
BUFFER_LIMIT   = 1_000_000 # bytes; drop chunk if upstream buffer exceeds this
CONNECT_TIMEOUT = 10       # seconds to wait for upstream WS open

# ── TLS / SSL context ─────────────────────────────────────────────────────────
# Python on macOS does not use the system keychain by default; the bundled
# certifi CA store guarantees the Sarvam and Deepgram TLS certificates can
# always be verified regardless of the host OS certificate configuration.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# ── Upstream URL builders ──────────────────────────────────────────────────────

def _sarvam_ws_url(language_code: str = "en-IN") -> str:
    """Build the Sarvam streaming STT WebSocket URL with VAD parameters.

    Model history:
        saarika:v2       — DEPRECATED (server closes with code 4000 immediately)
        saarika:v2.5     — Legacy (still accepted; no mode parameter)
        saaras:v2.5      — Legacy (still accepted; no mode parameter)
        saaras:v3        — Current recommended model (requires explicit mode=)
    Ref: https://docs.sarvam.ai/api-reference-docs/getting-started/models/saaras
         https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/streaming-api
    TODO: if you see close code 4000 again, check the above URL for a newer model ID.
    """
    from urllib.parse import urlencode
    params = urlencode({
        "model":               "saaras:v3",   # current recommended (was saarika:v2, now deprecated)
        "mode":                "transcribe",   # required for saaras:v3; transcribe = original language
        "language_code":       language_code,
        "sample_rate":         "16000",
        "high_vad_sensitivity": "true",
        "vad_signals":         "true",
        # NOTE: do NOT add input_audio_codec here — saaras:v3 requires JSON text
        # frames with base64-encoded audio, NOT raw binary WebSocket frames.
        # See: docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe/ws
    })
    return f"wss://api.sarvam.ai/speech-to-text/ws?{params}"


def _deepgram_ws_url(language_code: str = "en") -> str:
    """Build the Deepgram streaming STT WebSocket URL."""
    from urllib.parse import urlencode
    # Normalise locale tags like "en-IN" → "en" (Deepgram uses short codes)
    lang = language_code.split("-")[0]
    params = urlencode({
        "model":           "nova-3",
        "language":        lang,
        "encoding":        "linear16",
        "sample_rate":     "16000",
        "endpointing":     "500",
        "interim_results": "true",
        "punctuate":       "true",
        "smart_format":    "true",
    })
    return f"wss://api.deepgram.com/v1/listen?{params}"


# ── Upstream connector ─────────────────────────────────────────────────────────

async def _open_upstream(language_code: str):
    """Try Sarvam first, fall back to Deepgram.

    Returns (websocket, provider_name) or raises RuntimeError if both fail.

    Safety net: after a successful TCP/TLS handshake, Sarvam may immediately
    send a 4xxx close frame (e.g. deprecated model, auth failure, quota
    exhausted).  We peek at the first message with a short timeout; if the
    socket closes with a 4xxx application code before we receive any transcript
    frame, we treat it as a Sarvam-side failure and fall through to Deepgram.
    """
    EARLY_DISCONNECT_TIMEOUT = 2.0   # seconds to wait for the first server frame

    # --- Sarvam ---
    try:
        url     = _sarvam_ws_url(language_code)
        headers = {"api-subscription-key": settings.SARVAM_API_KEY}
        ws = await asyncio.wait_for(
            websockets.connect(url, additional_headers=headers, ssl=_SSL_CONTEXT),
            timeout=CONNECT_TIMEOUT,
        )
        logger.info("stt_stream | upstream=Sarvam connected")

        # Peek at the first frame: if Sarvam sends a 4xxx close within
        # EARLY_DISCONNECT_TIMEOUT seconds this is a logical rejection
        # (deprecated model, bad API key, quota exceeded, etc.)
        try:
            first_raw = await asyncio.wait_for(
                ws.recv(), timeout=EARLY_DISCONNECT_TIMEOUT
            )
            # Check if the first frame is a Sarvam error message (e.g. wrong
            # audio format, invalid model, etc.) — fall through to Deepgram.
            try:
                first_json = json.loads(first_raw) if isinstance(first_raw, (str, bytes)) else {}
            except Exception:
                first_json = {}
            if first_json.get("type") == "error":
                err_msg = (first_json.get("data") or {}).get("message", str(first_raw))
                logger.warning(
                    "stt_stream | Sarvam returned error frame immediately: %s — falling back to Deepgram",
                    err_msg,
                )
                try:
                    await ws.close()
                except Exception:
                    pass
            else:
                # Good first frame — store it and return the connection
                ws._stt_peeked_frame = first_raw
                logger.debug("stt_stream | Sarvam first frame (ok): %.200s", first_raw)
                return ws, "sarvam"
        except websockets.ConnectionClosedError as exc:
            code = exc.code if exc.code is not None else 0
            logger.warning(
                "stt_stream | Sarvam closed immediately after connect: "
                "code=%s reason=%s — falling back to Deepgram", code, exc.reason
            )
            if 4000 <= code <= 4999:
                # Application-layer rejection — fall through to Deepgram
                pass
            else:
                raise   # unexpected close code; propagate so caller sees it
        except asyncio.TimeoutError:
            # No frame within 2 s but socket still open — perfectly normal,
            # return the connection and start streaming.
            ws._stt_peeked_frame = None
            return ws, "sarvam"

    except websockets.ConnectionClosedError:
        # Re-raised from the 4xxx branch above; fall through to Deepgram
        pass
    except Exception as exc:
        logger.warning("stt_stream | Sarvam upstream failed: %s — trying Deepgram", exc)

    # --- Deepgram fallback ---
    url     = _deepgram_ws_url(language_code)
    headers = {"Authorization": f"Token {settings.DEEPGRAM_API_KEY}"}
    ws = await asyncio.wait_for(
        websockets.connect(url, additional_headers=headers, ssl=_SSL_CONTEXT),
        timeout=CONNECT_TIMEOUT,
    )
    ws._stt_peeked_frame = None
    logger.info("stt_stream | upstream=Deepgram connected")
    return ws, "deepgram"


# ── Frame normaliser ───────────────────────────────────────────────────────────

def _normalise_frame(raw, provider: str) -> Optional[dict]:
    """Convert an upstream raw frame into our canonical event dict.

    Returns None for frames that should be silently dropped (e.g. keep-alives
    or metadata frames that don't contain transcript data).
    """
    # Parse if it is a string / bytes JSON blob
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Non-JSON text (e.g. pong frames) — drop silently
        return None

    if provider == "sarvam":
        # Log every raw frame until transcript flow is confirmed (lower to DEBUG later).
        logger.info("stt_stream | RAW SARVAM FRAME: %s", raw)

        frame_type = data.get("type", "")

        # Sarvam v3 error frame: {"type":"error","data":{"message":"..."}}
        if frame_type == "error":
            err_msg = (data.get("data") or {}).get("message", str(data))
            logger.error("stt_stream | Sarvam error frame: %s", err_msg)
            return {"type": "stt_error", "message": err_msg}

        # Sarvam v3 transcript frame: {"type":"data","data":{"transcript":...,"metrics":{...}}}
        if frame_type == "data":
            payload    = data.get("data", {})
            transcript = payload.get("transcript", "")
            # v3 doesn't send is_final; every "data" frame is a final segment
            if not transcript:
                return None
            return {
                "type":       "stt_final",
                "transcript": transcript,
                "is_final":   True,
            }

        # VAD signal frames: {"type":"events","data":{"signal_type":"START_SPEECH"/"END_SPEECH"}}
        if frame_type == "events":
            signal = (data.get("data") or {}).get("signal_type", "")
            if signal == "START_SPEECH":
                # Forward as a synthetic stt_interim with an ellipsis so the
                # textarea immediately shows "…" while the user is speaking —
                # Sarvam v3 only sends a final transcript on END_SPEECH, so
                # without this there is zero visual feedback during speech.
                return {"type": "stt_interim", "transcript": "…", "is_final": False}
            # END_SPEECH — drop silently, the real transcript follows immediately
            return None

        # Legacy / fallback: FLAT shape {"transcript":..., "is_final":...}
        transcript = data.get("transcript", "")
        is_final   = data.get("is_final", False)
        if not transcript:
            return None
        return {
            "type":       "stt_final" if is_final else "stt_interim",
            "transcript": transcript,
            "is_final":   is_final,
        }

    if provider == "deepgram":
        # Deepgram nested path: results.channels[0].alternatives[0].transcript
        try:
            channel   = data["channel"]["alternatives"][0]
            transcript = channel.get("transcript", "")
            is_final   = data.get("is_final", False)
        except (KeyError, IndexError, TypeError):
            return None
        if not transcript:
            return None
        return {
            "type":       "stt_final" if is_final else "stt_interim",
            "transcript": transcript,
            "is_final":   is_final,
        }

    return None


# ── WebSocket endpoint handler ─────────────────────────────────────────────────

async def stt_stream_endpoint(client: WebSocket):
    """FastAPI WebSocket handler — registered in server.py as /voice/stream.

    One handler instance per connected browser tab. Upstream connections are
    created on demand (start_stt) and torn down immediately on stop or disconnect,
    so idle browser tabs don't hold open upstream sockets against the Sarvam quota.

    A client can call start_stt multiple times within the same WebSocket connection
    to restart recording after a pause — no reconnect needed.
    """
    await client.accept()
    logger.info("stt_stream | client connected")

    upstream: Optional[websockets.ClientConnection] = None
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
        logger.info("stt_stream | upstream torn down (%s)", reason)

    async def heartbeat():
        """Ping the upstream WS every PING_INTERVAL seconds."""
        while True:
            await asyncio.sleep(PING_INTERVAL)
            # close_code is None while the connection is open; set to an int once closed.
            # This avoids importing the State enum which can be a different object
            # at runtime depending on the websockets internal import path.
            if upstream and upstream.close_code is None:
                try:
                    await upstream.ping()
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception:
                    break

    async def forward_upstream_frames():
        """Receive frames from upstream and relay them to the browser client."""
        try:
            # If _open_upstream() peeked the first frame to check for an early
            # disconnect, replay it now before entering the normal receive loop.
            peeked = getattr(upstream, "_stt_peeked_frame", None)
            if peeked is not None:
                event = _normalise_frame(peeked, provider)
                if event:
                    await client.send_text(json.dumps(event))

            async for raw in upstream:
                event = _normalise_frame(raw, provider)
                if event:
                    await client.send_text(json.dumps(event))
        except websockets.ConnectionClosed as exc:
            logger.info("stt_stream | upstream closed: code=%s reason=%s", exc.code, exc.reason)
            await client.send_text(json.dumps({
                "type":   "stt_disconnected",
                "code":   exc.code,
                "reason": exc.reason or "",
            }))
        except Exception as exc:
            logger.error("stt_stream | upstream frame relay error: %s", exc)
            await client.send_text(json.dumps({
                "type":    "stt_error",
                "message": str(exc),
            }))

    # ── main receive loop ──────────────────────────────────────────────────────

    try:
        while True:
            # Receive either a text control frame or a binary audio chunk
            message = await client.receive()

            # ── Binary audio chunk ─────────────────────────────────────────────
            if "bytes" in message and message["bytes"]:
                if upstream and upstream.close_code is None:
                    # One-time debug log to confirm state at runtime.
                    if not getattr(upstream, "_stt_state_logged", False):
                        logger.info(
                            "stt_stream | upstream.state=%r type=%s close_code=%r",
                            upstream.state, type(upstream.state).__name__, upstream.close_code,
                        )
                        upstream._stt_state_logged = True
                    # Back-pressure guard
                    buffered = getattr(upstream.transport, "get_write_buffer_size", lambda: 0)()
                    if buffered < BUFFER_LIMIT:
                        try:
                            if provider == "sarvam":
                                # saaras:v3 requires JSON text frames with
                                # base64-encoded audio — raw binary is rejected.
                                audio_b64 = base64.b64encode(message["bytes"]).decode("ascii")
                                frame = json.dumps({
                                    "audio": {
                                        "data":        audio_b64,
                                        "sample_rate": "16000",
                                        "encoding":    "audio/wav",
                                    }
                                })
                                await upstream.send(frame)
                            else:
                                # Deepgram accepts raw binary PCM directly.
                                await upstream.send(message["bytes"])
                        except websockets.exceptions.ConnectionClosed as exc:
                            logger.warning("stt_stream | upstream closed while sending chunk: %s", exc)
                        except Exception as exc:
                            logger.warning("stt_stream | upstream send failed: %s", exc)
                    else:
                        logger.debug("stt_stream | back-pressure: chunk dropped (buffer=%d)", buffered)
                else:
                    await client.send_text(json.dumps({
                        "type":    "stt_error",
                        "message": "STT not initialised — send start_stt first",
                    }))
                continue

            # ── Text / JSON control frame ──────────────────────────────────────
            if "text" not in message or not message["text"]:
                continue

            try:
                ctrl = json.loads(message["text"])
            except json.JSONDecodeError:
                continue

            ctrl_type = ctrl.get("type", "")

            # --- start_stt ---
            if ctrl_type == "start_stt":
                # Tear down any existing session cleanly
                if upstream:
                    await teardown("restart")

                lang = ctrl.get("language_code", "en-IN")
                try:
                    upstream, provider = await _open_upstream(lang)
                except Exception as exc:
                    logger.error("stt_stream | could not open upstream: %s", exc)
                    await client.send_text(json.dumps({
                        "type":    "stt_error",
                        "message": f"Could not connect to STT provider: {exc}",
                    }))
                    continue

                # Debug: log state immediately after connect so we can confirm
                # the runtime enum type for future reference (safe to remove later).
                logger.info(
                    "stt_stream | upstream ready | state=%r close_code=%r provider=%s",
                    upstream.state, upstream.close_code, provider,
                )

                # Start parallel tasks: heartbeat + frame forwarding
                ping_task = asyncio.create_task(heartbeat())
                fwd_task  = asyncio.create_task(forward_upstream_frames())

                def _fwd_done(t: asyncio.Task):
                    exc = t.exception() if not t.cancelled() else None
                    if exc:
                        logger.error("stt_stream | forward_upstream_frames crashed: %s", exc)
                fwd_task.add_done_callback(_fwd_done)

                await client.send_text(json.dumps({"type": "stt_ready"}))
                logger.info(
                    "stt_stream | session started | provider=%s lang=%s",
                    provider, lang,
                )

            # --- stop_stt ---
            elif ctrl_type == "stop_stt":
                await teardown("client-stop")
                await client.send_text(json.dumps({"type": "stt_stopped"}))

    except WebSocketDisconnect:
        logger.info("stt_stream | client disconnected")
    except RuntimeError as exc:
        # Swallow the "Cannot call receive once a disconnect message has been
        # received" error — happens when the client closes the WS while we're
        # mid-receive; it's benign but noisy.
        if "disconnect" in str(exc).lower():
            logger.info("stt_stream | client WS closed cleanly mid-receive")
        else:
            logger.error("stt_stream | unexpected error: %s", exc)
    except Exception as exc:
        logger.error("stt_stream | unexpected error: %s", exc)
    finally:
        await teardown("connection-end")
