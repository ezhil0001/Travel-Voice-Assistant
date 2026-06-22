# Travel Voice Assistant

A production-grade, multi-agent travel assistant that responds to both typed and spoken queries. Built with LangChain, LangGraph, FastAPI, and Angular 19. The system routes each query to one or more specialised AI agents (weather, flights, attractions, currency, timezone), produces richly structured responses for the UI, and synthesises speech that is always concise and natural.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Application Workflow](#application-workflow)
- [Voice Interaction Flow](#voice-interaction-flow)
- [WebSocket Communication](#websocket-communication)
- [Dual-View Response System](#dual-view-response-system)
- [Retry and Fallback Behaviour](#retry-and-fallback-behaviour)
- [Context Retention](#context-retention)
- [Middleware Pipeline](#middleware-pipeline)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Key Features

| Feature | Detail |
|---|---|
| **Multi-intent routing** | A single query like "weather in Tokyo, flights from Chennai, and currency" triggers all three agents in parallel via a single LLM dispatch |
| **Voice I/O** | Real-time streaming STT (WebSocket) + REST TTS with provider-level retry and automatic fallback |
| **Dual-view responses** | Every assistant message has an **Optimised View** (rich markdown tables/cards) and a **Summary View** (2-4 sentence plain text) generated from the same pipeline run |
| **TTS speaks Summary text** | The speech output always uses the concise Summary text — never reads markdown tables or structured layouts aloud |
| **Context retention** | Pronouns like "there" or "that city" are automatically resolved to the last mentioned destination from conversation history before any agent is called |
| **LLM resilience** | Ollama (local, primary) → Groq cloud (automatic fallback); three retry attempts on Ollama before switching |
| **Session memory** | Per-session conversation history (capped at 10 turns) injected into every prompt |
| **Auto-play welcome** | On page load the welcome message is synthesised and played automatically |
| **95 backend tests** | Full unit test coverage across agents, middleware, graph, voice pipeline, and server endpoints |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Angular 19 Frontend               │
│  ┌──────────────┐  ┌──────────────────────────────┐ │
│  │  Chat Window │  │       Voice Panel            │ │
│  │  (text input)│  │  (mic → VAD → WebSocket STT) │ │
│  └──────┬───────┘  └──────────────┬───────────────┘ │
│         │ POST /text/query         │ WS /voice/stream │
│         │ POST /voice/query        │ POST /voice/query│
└─────────┼──────────────────────────┼─────────────────┘
          │                          │
┌─────────▼──────────────────────────▼─────────────────┐
│                  FastAPI Server (server.py)           │
│  ┌──────────────────────────────────────────────────┐ │
│  │              LangGraph Pipeline                  │ │
│  │                                                  │ │
│  │  pre_middleware → supervisor → run_agents        │ │
│  │       → merge_responses → post_middleware        │ │
│  │       → summarize_response                       │ │
│  │                                                  │ │
│  │  Agents: Weather · Flight · Attractions          │ │
│  │          Currency · Timezone · General LLM       │ │
│  └──────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────┐ │
│  │              Voice Layer                         │ │
│  │  STT: Sarvam (primary) → Deepgram (fallback)    │ │
│  │  TTS: Sarvam (primary) → Deepgram (fallback)    │ │
│  └──────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────┐ │
│  │              LLM Layer (ModelLayer)              │ │
│  │  Ollama (local, primary, 3 retries)              │ │
│  │  → Groq cloud (automatic fallback)               │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
          │                          │
    External APIs             External APIs
  OpenWeatherMap            Sarvam AI (STT/TTS)
  FlightAPI.io              Deepgram (STT/TTS)
  GeoNames                  Groq (LLM)
  ExchangeRate-API
  TimeAPI.io
```

---

## Tech Stack

### Backend

| Layer | Technology |
|---|---|
| API server | FastAPI + Uvicorn |
| AI orchestration | LangGraph (StateGraph), LangChain |
| Primary LLM | Ollama (`qwen3.5:122b` or any local model) |
| Fallback LLM | Groq (`openai/gpt-oss-120b` or any Groq model) |
| STT (primary) | Sarvam AI `saarika:v2.5` |
| STT (fallback) | Deepgram |
| TTS (primary) | Sarvam AI `bulbul:v3` (speaker: `ritu`) |
| TTS (fallback) | Deepgram |
| Weather data | OpenWeatherMap API |
| Flight data | FlightAPI.io |
| Attractions data | GeoNames API |
| Currency data | ExchangeRate-API |
| Timezone data | TimeAPI.io |
| Language | Python 3.11+ |

### Frontend

| Layer | Technology |
|---|---|
| Framework | Angular 19 |
| Language | TypeScript 5.7 |
| State management | RxJS BehaviorSubjects via `ChatStateService` |
| Markdown rendering | `marked` library via `MarkdownPipe` |
| Audio capture | Web Audio API + `MediaRecorder` |
| Voice activity detection | Custom RMS-based VAD |
| Real-time STT | WebSocket (`/voice/stream`) with PCM streaming |
| HTTP client | Angular `HttpClient` with session interceptor |
| Styling | SCSS |
| Testing | Karma + Jasmine |

---

## Folder Structure

```
Travel-Assistance/
│
├── server.py                  # FastAPI entry point — all HTTP/WS endpoints
├── requirements.txt           # Python dependencies
├── .env                       # Local secrets (git-ignored)
├── .env.example               # Template — copy to .env and fill in keys
│
├── agents/                    # Sub-agents (one per domain)
│   ├── base_agent.py          # Shared chain construction, tool binding, retry
│   ├── model_layer.py         # LLM abstraction: Ollama → Groq fallback
│   ├── supervisor_agent.py    # Intent detection (LLM-based, returns list)
│   ├── weather_agent.py
│   ├── flight_agent.py
│   ├── attractions_agent.py
│   ├── currency_agent.py
│   └── timezone_agent.py
│
├── graph/
│   └── travel_graph.py        # LangGraph StateGraph — 6 nodes, public run_graph_full()
│
├── middleware/
│   ├── pre_model.py           # Clean STT input, resolve pronouns, inject history
│   ├── post_model.py          # Strip markdown/URLs, truncate for TTS
│   ├── dynamic_prompt.py      # Per-turn prompt enrichment
│   └── pipeline.py            # Middleware orchestration
│
├── voice/
│   ├── stt.py                 # STT orchestrator: Sarvam → Deepgram
│   ├── tts.py                 # TTS orchestrator: Sarvam → Deepgram
│   ├── stt_stream.py          # WebSocket STT proxy (/voice/stream)
│   └── tts_stream.py          # WebSocket TTS proxy (/tts/stream)
│
├── tools/                     # LangChain tools (one per API)
│   ├── weather_tool.py        # OpenWeatherMap
│   ├── flight_tool.py         # FlightAPI.io
│   ├── attractions_tool.py    # GeoNames
│   ├── currency_tool.py       # ExchangeRate-API
│   └── timezone_tool.py       # TimeAPI.io
│
├── state/
│   └── schema.py              # TravelState TypedDict shared by all graph nodes
│
├── config/
│   ├── settings.py            # Pydantic settings — reads .env
│   ├── prompts.json           # All prompt templates (no prompts in Python files)
│   └── cities.json            # City list for pre-middleware city detection
│
├── tests/                     # 95 backend unit tests
│   ├── test_server_endpoints.py
│   ├── test_graph_routing.py
│   ├── test_multi_intent.py
│   ├── test_middleware_pipeline.py
│   ├── test_agent_routing.py
│   ├── test_api_tools.py
│   ├── test_model_routing.py
│   ├── test_voice_pipeline.py
│   ├── test_config_and_env.py
│   └── test_integration.py    # Live integration (requires running server)
│
└── travel-frontend/           # Angular 19 SPA
    └── src/app/
        ├── api/               # HTTP/WS service wrappers
        │   ├── text-api.service.ts
        │   ├── voice-api.service.ts
        │   └── stt-stream.service.ts
        ├── core/
        │   ├── services/      # ChatStateService, SessionService, LoggerService
        │   └── interceptors/  # Session ID header injection
        ├── features/chat/
        │   └── components/
        │       ├── chat-window/      # Text input, message list
        │       ├── message-bubble/   # Optimised/Summary tabs, markdown rendering
        │       └── voice-panel/      # Mic button, VAD, audio pipeline
        ├── shared/
        │   ├── components/
        │   │   ├── tool-activity/    # Collapsible data-source panel
        │   │   ├── audio-wave/       # Animated wave (red = listening, green = speaking)
        │   │   └── mic-button/
        │   └── pipes/
        │       └── markdown.pipe.ts  # marked-based markdown → SafeHtml
        └── models/
            ├── message.model.ts
            └── voice-state.model.ts
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 also works |
| Node.js | 18+ | LTS recommended |
| npm | 9+ | Bundled with Node |
| Angular CLI | 19.x | `npm install -g @angular/cli` |
| Ollama | latest | Optional but recommended for local LLM |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ezhil0001/Travel-Voice-Assistant.git
cd Travel-Voice-Assistant
```

### 2. Set up the Python environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in all required API keys (see [Environment Variables](#environment-variables) below).

### 4. Install Angular dependencies

```bash
cd travel-frontend
npm install
cd ..
```

### 5. Pull the Ollama model (optional but recommended)

```bash
ollama pull qwen3.5:122b
```

If Ollama is not available, all LLM calls automatically fall back to Groq.

---

## Environment Variables

Copy `.env.example` to `.env` and configure every key before starting the server.

### External API Keys

| Variable | Service | Sign-up URL | Required |
|---|---|---|---|
| `OPENWEATHER_API_KEY` | OpenWeatherMap | https://openweathermap.org/api | ✅ |
| `FLIGHTAPI_KEY` | FlightAPI.io | https://flightapi.io | ✅ |
| `GEONAMES_USERNAME` | GeoNames | https://www.geonames.org/login | ✅ |
| `EXCHANGERATE_API_KEY` | ExchangeRate-API | https://www.exchangerate-api.com | ✅ |

> **GeoNames note:** After registering, you must also visit  
> https://www.geonames.org/manageaccount and click **Enable Free Webservices**.

### Voice Layer

| Variable | Service | Notes |
|---|---|---|
| `SARVAM_API_KEY` | Sarvam AI | Primary STT (`saarika:v2.5`) and TTS (`bulbul:v3`) |
| `DEEPGRAM_API_KEY` | Deepgram | Fallback STT and TTS |

Both keys are required. If Sarvam fails after retries, the system automatically switches to Deepgram without user-visible interruption.

### LLM Layer

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | Cloud LLM fallback. Sign up at https://console.groq.com |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Any model available in your Groq account |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL of your Ollama instance |
| `OLLAMA_MODEL` | `qwen3.5:122b` | Any model you have pulled locally |
| `OLLAMA_API_KEY` | — | Only required for remote/hosted Ollama endpoints |

### Middleware Tuning

These values control runtime behaviour without touching Python code:

| Variable | Default | Effect |
|---|---|---|
| `PRE_MODEL_HISTORY_WINDOW` | `3` | Number of past conversation turns injected into each prompt |
| `POST_MODEL_MAX_CHARS` | `800` | Maximum characters in the TTS-bound response before truncation |
| `POST_MODEL_OVERFLOW_SUFFIX` | `...for more details, just ask` | Appended when a response is truncated |
| `VOICE_MAX_RETRIES` | `2` | Sarvam retry attempts before falling back to Deepgram |

---

## Running the Application

### Start the backend

```bash
# From the project root, with .venv activated
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`.  
Interactive API docs: `http://localhost:8000/docs`

### Start the frontend

```bash
cd travel-frontend
npx ng serve
```

The Angular app will be available at `http://localhost:4200`.

> **Important:** The backend must be running before the frontend will function. The Angular dev server proxies nothing — it calls `http://localhost:8000` directly.

### Start Ollama (if using local LLM)

```bash
ollama serve
```

Run this in a separate terminal before starting the backend. If Ollama is not running, all LLM calls fall back to Groq automatically.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/text/query` | Text-only pipeline — returns JSON with optimised content, summary, and tool events |
| `POST` | `/voice/query` | Full voice pipeline — STT → graph → TTS → returns JSON with audio and all response fields |
| `POST` | `/tts/synthesize` | Synthesise arbitrary text to speech (used for auto-played welcome message) |
| `WS` | `/voice/stream` | Real-time PCM streaming STT — proxies to Sarvam or Deepgram |
| `WS` | `/tts/stream` | Real-time streaming TTS |
| `GET` | `/health` | Liveness check — reports Ollama and Sarvam connectivity |

### `/text/query` — Request

```json
{
  "text": "What is the weather in Tokyo?",
  "session_id": "my-session-id"
}
```

### `/text/query` — Response

```json
{
  "response": "TTS-ready merged text (plain, no markdown)",
  "summary_response": "Short 2-4 sentence conversational summary",
  "intent": "weather",
  "intents": ["weather"],
  "agent_responses": {
    "weather": "**Tokyo — Current Conditions**\n| 🌡️ Temperature | 28°C ..."
  },
  "tool_events": [
    {
      "tool_name": "get_weather",
      "label": "Fetching weather data",
      "status": "success",
      "detail": "What is the current weather in Tokyo?",
      "duration_ms": 412,
      "source": "OpenWeatherMap API"
    }
  ],
  "session_id": "my-session-id"
}
```

### `/voice/query` — Request

`multipart/form-data` with:
- `audio_file`: WAV or WebM audio blob
- `X-Session-Id` header: session identifier

### `/health` — Response

```json
{
  "status": "ok",
  "timestamp": "2026-06-23T10:00:00",
  "ollama": true,
  "sarvam": true
}
```

---

## Application Workflow

### Text query flow

```
User types message
    │
    ▼
ChatWindowComponent.sendText()
    │
    ▼
POST /text/query
    │
    ▼
LangGraph pipeline (6 nodes)
    │
    ▼
JSON response { response, summary_response, agent_responses, tool_events }
    │
    ▼
Message bubble updated
    ├── Optimised tab → structured markdown cards (from agent_responses)
    └── Summary tab  → plain conversational text (from summary_response)
```

### Voice query flow

```
User presses mic button
    │
    ▼
getUserMedia() → AudioContext
    │
    ├── AnalyserNode     → VAD (RMS silence detection)
    └── ScriptProcessor  → PCM downsampled to 16kHz
           │
           ▼
    WebSocket /voice/stream (real-time STT)
           │
           ▼
    stt_interim / stt_final events → live transcript bubble
           │
    3s silence detected by VAD
           │
           ▼
    MediaRecorder blob + committed transcript
           │
           ▼
    POST /voice/query
           │
           ▼
    STT (Sarvam → Deepgram fallback)
           │
           ▼
    LangGraph pipeline (same 6 nodes as text)
           │
           ▼
    TTS synthesises summary_response (never the optimised markdown)
           │
           ▼
    JSON { audio_base64, response, summary_response, agent_responses, ... }
           │
           ▼
    HTMLAudioElement plays WAV
    Message bubble updated (Optimised + Summary tabs)
```

---

## Voice Interaction Flow

### Microphone pipeline (per turn)

1. **`getUserMedia()`** — requests microphone access.
2. **`AudioContext`** — shared source node for both VAD and PCM streaming.
3. **`AnalyserNode`** — computes RMS energy every 100 ms. Speech begins when RMS ≥ 12.
4. **`ScriptProcessorNode`** — downsamples from browser native rate (44100/48000 Hz) to 16000 Hz Int16 PCM and sends binary frames over the WebSocket.
5. **`MediaRecorder`** — records the full audio blob in parallel for the `POST /voice/query` call.
6. **VAD grace window** — after 3 seconds of continuous silence following detected speech, `stopAndSend()` is called automatically.

### Live transcript

- `stt_interim` events update the user's message bubble in real time as words are spoken.
- `stt_final` events commit each segment to the accumulated transcript.
- On `stopAndSend()`, the bubble is sealed (cursor removed) and the full audio blob is posted.

### Continuous conversation mode

When the user starts a conversation session (mic button), after each assistant response finishes playing:

1. TTS `audio.onended` fires.
2. `startListening()` is called automatically.
3. The mic reopens for the next turn without any button press.

The user can press **End Session** or **Skip & Listen** at any time.

---

## WebSocket Communication

### `/voice/stream` — Streaming STT

```
Browser                           FastAPI server                    Sarvam / Deepgram
   │                                     │                                 │
   │── JSON { type: "start_stt",  ───────▶│                                 │
   │         session_id, language }       │── open upstream WS ────────────▶│
   │                                     │◀── connected ───────────────────│
   │◀── JSON { type: "stt_ready" } ───────│                                 │
   │                                     │                                 │
   │── binary PCM chunk ────────────────▶│── binary PCM chunk ────────────▶│
   │── binary PCM chunk ────────────────▶│── binary PCM chunk ────────────▶│
   │                                     │◀── transcript frame ────────────│
   │◀── JSON { type: "stt_interim" } ────│                                 │
   │◀── JSON { type: "stt_final"   } ────│                                 │
   │                                     │                                 │
   │── JSON { type: "stop_stt" } ───────▶│── close upstream ──────────────▶│
```

Audio frames sent before `stt_ready` are buffered (up to 200 frames) and flushed once the upstream connection is established.

---

## Dual-View Response System

Every assistant response has two views generated from a single pipeline run:

### Optimised View

- Generated by each sub-agent's `_DISPLAY_PROMPT` — rich markdown with tables, schedules, and structured data.
- Displayed in the UI using the `MarkdownPipe` (via `marked`) within `.markdown-body` styled sections.
- **Never spoken aloud.**

### Summary View

- Generated by the `summarize_response_node` — calls the LLM with the optimised content and a summarisation prompt.
- Always 2-4 plain sentences with no markdown, no tables, no structured layouts.
- Displayed in the Summary tab of every message bubble.
- **Always used for TTS speech synthesis.**

This guarantees that both views are in sync — the summary is derived directly from the optimised content, not generated independently.

```
agent_responses (rich markdown)
    │
    ├──▶ Optimised View (UI only, never spoken)
    │
    └──▶ summarize_response_node (LLM call)
              │
              ▼
         summary_response (2-4 sentences)
              │
              ├──▶ Summary View (UI)
              └──▶ TTS synthesis (spoken aloud)
```

---

## Retry and Fallback Behaviour

### LLM layer

| Scenario | Behaviour |
|---|---|
| Ollama responds slowly | `with_retry()` retries up to 3 times with exponential backoff |
| Ollama is unreachable | After 3 failed attempts, `with_fallbacks()` switches to Groq automatically |
| Groq also fails | `ModelLayer.invoke()` returns an error string — the graph node catches it |

### STT / TTS voice layer

| Scenario | Behaviour |
|---|---|
| Sarvam times out or returns an error | `retry_call()` retries up to `VOICE_MAX_RETRIES` times |
| All Sarvam retries exhausted | Deepgram is called transparently |
| Deepgram also fails | The endpoint returns an appropriate HTTP error to the frontend |

### Flight API

The FlightAPI.io endpoint occasionally returns HTTP 400 on the first request. The flight tool has its own retry loop (up to 5 attempts, 1.5s delay) before propagating the error.

---

## Context Retention

The `PreModelMiddleware` resolves location pronouns before any agent is dispatched:

| User says | History contains | Resolved to |
|---|---|---|
| "What time is it **there**?" | "I want to visit Tokyo" | "What time is it **in Tokyo**?" |
| "Is **that city** expensive?" | "Tell me about Paris" | "Is **Paris** expensive?" |
| "How's the weather in Rome?" | anything | Unchanged (explicit city present) |

The middleware scans both user and assistant turns in conversation history (most-recent-first) for the last mentioned city name. Substitution only triggers when no city is already present in the current query.

---

## Middleware Pipeline

```
Raw user input
    │
    ▼
PreModelMiddleware
    ├── Collapse STT whitespace artefacts
    ├── Resolve location pronouns ("there" → "in Tokyo")
    ├── Detect city for metadata
    └── Slice last N history turns for context
    │
    ▼
SupervisorAgent (intent detection)
    └── LLM returns JSON array of intents, e.g. ["weather", "timezone"]
    │
    ▼
run_agents_node
    ├── Single LLM call builds focused sub-queries for all intents
    ├── Each agent called with its focused query
    └── agent_responses dict populated
    │
    ▼
merge_responses_node
    ├── Single intent → returned directly, no LLM merge
    └── Multiple intents → LLM weaves them into one spoken paragraph
    │
    ▼
PostModelMiddleware
    ├── Strip markdown, URLs, asterisks
    └── Truncate to POST_MODEL_MAX_CHARS
    │
    ▼
summarize_response_node
    └── LLM generates 2-4 sentence plain-text summary from agent_responses
```

---

## Configuration Reference

All prompt templates live in `config/prompts.json` — no prompts are hardcoded in Python files. To change any prompt behaviour, edit the JSON file and restart the server.

| Key | Purpose |
|---|---|
| `supervisor_routing_prompt` | Intent detection — defines all five domains and examples |
| `batch_focused_query_prompt` | Extracts focused sub-queries for all detected intents in one LLM call |
| `merge_responses_prompt` | Merges multiple agent responses into one spoken paragraph |
| `summary_prompt` | Summarises the optimised content into 2-4 plain sentences for TTS |
| `city_to_iata` | City name → IATA airport code lookup (e.g. `"tokyo": "NRT"`) |
| `city_to_timezone` | City name → IANA timezone string (e.g. `"tokyo": "Asia/Tokyo"`) |

To add a new city to both lookups, edit `config/prompts.json` — no Python changes needed.

---

## Testing

### Run all backend tests (excluding live integration)

```bash
# From the project root with .venv activated
python -m pytest tests/ -q --ignore=tests/test_integration.py
```

Expected: **95 tests pass**.

### Run a specific test file

```bash
python -m pytest tests/test_graph_routing.py -v
```

### Run live integration tests (requires running server + valid API keys)

```bash
python -m pytest tests/test_integration.py -v
```

### Run Angular unit tests

```bash
cd travel-frontend
npx ng test --watch=false --browsers=ChromeHeadless
```

### Test coverage areas

| File | What it covers |
|---|---|
| `test_server_endpoints.py` | All HTTP endpoints including `/tts/synthesize` |
| `test_graph_routing.py` | Graph nodes: merge, summarize, post-middleware |
| `test_multi_intent.py` | Multi-intent routing and response merging |
| `test_middleware_pipeline.py` | Pre/post middleware, pronoun resolution |
| `test_agent_routing.py` | Supervisor intent detection |
| `test_api_tools.py` | Individual tool functions (weather, flight, etc.) |
| `test_model_routing.py` | ModelLayer Ollama → Groq fallback |
| `test_voice_pipeline.py` | STT/TTS provider retry and fallback |
| `test_config_and_env.py` | Settings loading from .env |

---

## Troubleshooting

### Backend won't start

**`ModuleNotFoundError`** — make sure the virtual environment is activated:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**`KeyError: 'GROQ_API_KEY'`** — `.env` file is missing or not in the project root. Copy `.env.example` to `.env` and fill in all keys.

---

### Ollama not responding

Check if Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

If it returns a connection error, start it:
```bash
ollama serve
```

If the model isn't downloaded:
```bash
ollama pull qwen3.5:122b
```

If you don't want to run Ollama at all, set `OLLAMA_BASE_URL` to a non-reachable URL (e.g. `http://localhost:1`). All calls will fail fast and fall through to Groq.

---

### No audio / TTS not playing on page load

Browsers block autoplay before the first user interaction. The welcome message audio will be silently skipped — the text message is always visible regardless. To hear the welcome greeting, interact with the page first (click anywhere) and then reload.

---

### Microphone not working in voice panel

1. Check that the browser has microphone permission for `localhost:4200`.
2. Check the browser console for `NotAllowedError` — this means permission was denied.
3. HTTPS is not required for `localhost`, but if running on a remote host you must serve over HTTPS for `getUserMedia()` to work.

---

### STT transcription is empty or wrong

1. Check `logs/test_results.log` for `STT | SarvamSTT` or `STT | DeepgramSTT` entries.
2. Verify `SARVAM_API_KEY` is set and active.
3. The STT provider expects 16kHz mono PCM. The frontend downsamples automatically — if you're testing with a raw audio file, ensure it matches this format.

---

### "Could not transcribe audio" (HTTP 422)

The STT returned an empty string. Common causes:
- Complete silence in the recording.
- Audio encoding issues (the frontend sends WebM; Sarvam accepts WebM — if you're POSTing manually, use the correct MIME type).
- `VOICE_MAX_RETRIES` exhausted and Deepgram also returned empty.

---

### Flight search returns no results

FlightAPI.io occasionally returns HTTP 400 on the first call. The tool retries up to 5 times automatically. If all retries fail:
1. Verify `FLIGHTAPI_KEY` in `.env`.
2. Check that both IATA codes are valid (the tool maps city names — check `city_to_iata` in `config/prompts.json`).
3. The requested date must be in the future.

---

### GeoNames attractions returning nothing

1. Verify `GEONAMES_USERNAME` is set.
2. Log in at https://www.geonames.org/manageaccount and confirm **Free Webservices** is enabled — this is a separate step from account registration.

---

### Angular build errors

```bash
cd travel-frontend
npm install          # ensure node_modules is populated
npx ng build --configuration development
```

If `marked` is missing:
```bash
npm install marked
```

---

### Checking logs

All backend activity is logged to `logs/test_results.log`:

```bash
tail -f logs/test_results.log
```

Useful patterns to grep for:

```bash
grep "STT result"          logs/test_results.log   # what was transcribed
grep "run_graph_full"      logs/test_results.log   # pipeline timings
grep "intent="             logs/test_results.log   # what intent was detected
grep "TTS synthesised"     logs/test_results.log   # TTS byte counts
grep "ERROR"               logs/test_results.log   # all errors
```
