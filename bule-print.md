# 🧭 Travel Voice Assistant — Complete Implementation Guide
> LangChain + LangGraph + Multi-Agent + Voice STT/TTS + Retell AI
> **Target: Complete in 2 Days | Git Commit Schedule Included**

---

## 📐 Architecture Overview

```
User Voice Input
      │
      ▼
[STT Layer]  ──── Primary: Sarvam STT
                   Fallback: Deepgram STT  (if Sarvam fails)
      │
      ▼
[LangChain Middleware Pipeline]
   ├── Pre-Model Middleware  (input validation, language detect, context inject)
   ├── Dynamic Prompt Builder
   ├── Supervisor Agent  (LangGraph)
   │      ├── Weather Sub-Agent
   │      ├── Flight Sub-Agent
   │      ├── Attractions Sub-Agent
   │      ├── Currency Sub-Agent
   │      └── Timezone Sub-Agent
   ├── Post-Model Middleware  (response formatting, safety check)
   └── Model Retry + Fallback  ──── Primary: Ollama (local)
                                     Fallback: Groq API
      │
      ▼
[TTS Layer]  ──── Primary: Sarvam TTS
                   Fallback: Deepgram TTS  (if Sarvam fails)
      │
      ▼
User hears spoken response
```

---

## 🗂️ Project Folder Structure

```
travel-voice-assistant/
├── agents/
│   ├── supervisor_agent.py
│   ├── weather_agent.py
│   ├── flight_agent.py
│   ├── attractions_agent.py
│   ├── currency_agent.py
│   └── timezone_agent.py
├── middleware/
│   ├── pre_model.py
│   ├── post_model.py
│   └── dynamic_prompt.py
├── voice/
│   ├── stt.py          # STT with Sarvam + Deepgram fallback
│   └── tts.py          # TTS with Sarvam + Deepgram fallback
├── tools/
│   ├── weather_tool.py
│   ├── flight_tool.py
│   ├── attractions_tool.py
│   ├── currency_tool.py
│   └── timezone_tool.py
├── graph/
│   └── travel_graph.py    # LangGraph state machine
├── config/
│   └── settings.py
├── tests/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   ├── test_phase4.py
│   ├── test_phase5.py
│   └── run_all_tests.py
├── logs/
│   └── test_results.log
├── server.py
├── requirements.txt
└── .env
```

---

## ⏱️ 2-Day Git Commit Schedule

### DAY 1 — Core Infrastructure

| Time | Phase | What to Build | Git Commit Message |
|------|-------|---------------|-------------------|
| 9:00 AM | Phase 1 | Project setup, env, config, folder structure | `feat: initial project scaffold and config setup` |
| 11:00 AM | Phase 2 | Tools (all 5 API integrations) + unit tests | `feat: add weather/flight/attractions/currency/timezone tools` |
| 1:00 PM | Phase 3 | Middleware (pre-model, post-model, dynamic prompt) | `feat: add langchain middleware pipeline pre and post model` |
| 3:00 PM | Phase 4 | Model layer — Ollama + Groq + retry/fallback logic | `feat: add model retry and fallback mechanism ollama to groq` |
| 5:00 PM | Phase 5 | Voice STT/TTS — Sarvam + Deepgram + fallback | `feat: add stt tts voice layer with sarvam and deepgram fallback` |

### DAY 2 — Agents + Graph + Server

| Time | Phase | What to Build | Git Commit Message |
|------|-------|---------------|-------------------|
| 9:00 AM | Phase 6 | Sub-agents (5 agents) | `feat: add sub-agents weather flight attractions currency timezone` |
| 11:00 AM | Phase 7 | Supervisor Agent + LangGraph state machine | `feat: add supervisor agent and langgraph travel state machine` |
| 1:00 PM | Phase 8 | Full integration — voice → graph → voice | `feat: integrate voice pipeline with langgraph multi-agent system` |
| 3:00 PM | Phase 9 | FastAPI server + all phase tests pass check | `feat: add fastapi server and run all phase tests with pass log` |
| 5:00 PM | Phase 10 | Bug fixes, cleanup, final test run | `fix: resolve integration issues and finalize all test cases` |

---

## 📦 Phase 1 — Project Setup

### `requirements.txt`
```
langchain==0.2.0
langgraph==0.1.0
langchain-community==0.2.0
langchain-groq==0.1.0
fastapi==0.111.0
uvicorn==0.29.0
httpx==0.27.0
python-dotenv==1.0.0
requests==2.31.0
pytest==8.2.0
ollama==0.2.0
groq==0.9.0
```

### `.env`
```
OPENWEATHER_API_KEY=your_key
AMADEUS_API_KEY=your_key
AMADEUS_API_SECRET=your_secret
OPENTRIPMAP_API_KEY=your_key
EXCHANGERATE_API_KEY=your_key
SARVAM_API_KEY=your_key
DEEPGRAM_API_KEY=your_key
GROQ_API_KEY=your_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

### `config/settings.py`
```python
import os
from dotenv import load_dotenv
load_dotenv()

OPENWEATHER_API_KEY   = os.getenv("OPENWEATHER_API_KEY")
AMADEUS_API_KEY       = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET    = os.getenv("AMADEUS_API_SECRET")
OPENTRIPMAP_API_KEY   = os.getenv("OPENTRIPMAP_API_KEY")
EXCHANGERATE_API_KEY  = os.getenv("EXCHANGERATE_API_KEY")
SARVAM_API_KEY        = os.getenv("SARVAM_API_KEY")
DEEPGRAM_API_KEY      = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL          = os.getenv("OLLAMA_MODEL", "llama3")
```

### Git Commit
```bash
git init
git add .
git commit -m "feat: initial project scaffold and config setup"
```

---

## 🔧 Phase 2 — Tools (API Integrations)

### AI Agent Prompt for Phase 2

```
You are a senior Python developer implementing API tool functions for a travel assistant.

Your task: Build 5 LangChain-compatible tool functions. Each tool must:
1. Accept typed parameters
2. Make an HTTP GET request to the specified API
3. Return a clean JSON-serializable dict
4. Handle HTTP errors and return {"error": "<message>"} on failure

Tools to implement:

--- TOOL 1: get_weather ---
API: OpenWeatherMap
URL: https://api.openweathermap.org/data/2.5/weather
Params: q=<city>, appid=<OPENWEATHER_API_KEY>, units=metric
Return: { city, temperature, feels_like, description, humidity, wind_speed }

--- TOOL 2: get_flights ---
API: Amadeus (use Test environment)
Flow: First POST to https://test.api.amadeus.com/v1/security/oauth2/token
      Then GET https://test.api.amadeus.com/v2/shopping/flight-offers
Params: originLocationCode, destinationLocationCode, departureDate, adults=1, max=3
Return: list of { airline, price, currency, stops, departure, arrival }

--- TOOL 3: get_attractions ---
API: OpenTripMap
URL: https://api.opentripmap.com/0.1/en/places/radius
Params: radius=5000, lon=<longitude>, lat=<latitude>, apikey=<key>
First geocode city using: https://api.opentripmap.com/0.1/en/places/geoname?name=<city>
Return: list of { name, kinds, distance }

--- TOOL 4: get_timezone ---
API: TimeAPI.io
URL: https://timeapi.io/api/Time/current/zone
Params: timeZone=<IANA_timezone e.g. Asia/Tokyo>
Return: { timezone, datetime, hour, minute }

--- TOOL 5: get_currency ---
API: ExchangeRate-API
URL: https://v6.exchangerate-api.com/v6/<EXCHANGERATE_API_KEY>/pair/<from>/<to>/<amount>
Return: { from_currency, to_currency, amount, result, rate }

Write each tool as a @tool decorated LangChain function in separate files:
tools/weather_tool.py, tools/flight_tool.py, tools/attractions_tool.py,
tools/timezone_tool.py, tools/currency_tool.py

Use config/settings.py for all API keys. Add proper docstrings to each tool.
```

### Test Case — `tests/test_phase2.py`
```python
"""Phase 2 Test: All 5 API Tools"""
import pytest
import logging

logging.basicConfig(filename="logs/test_results.log", level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

from tools.weather_tool import get_weather
from tools.flight_tool import get_flights
from tools.attractions_tool import get_attractions
from tools.timezone_tool import get_timezone
from tools.currency_tool import get_currency

def test_weather():
    result = get_weather.invoke({"city": "Tokyo"})
    assert "temperature" in result, f"FAIL: {result}"
    log.info("PASS | test_weather | Tokyo weather fetched")

def test_flights():
    result = get_flights.invoke({
        "origin": "JFK", "destination": "NRT",
        "date": "2025-03-15"
    })
    assert isinstance(result, list), f"FAIL: {result}"
    log.info("PASS | test_flights | JFK->NRT flights fetched")

def test_attractions():
    result = get_attractions.invoke({"city": "Paris"})
    assert isinstance(result, list), f"FAIL: {result}"
    log.info("PASS | test_attractions | Paris attractions fetched")

def test_timezone():
    result = get_timezone.invoke({"timezone": "Asia/Tokyo"})
    assert "datetime" in result, f"FAIL: {result}"
    log.info("PASS | test_timezone | Tokyo timezone fetched")

def test_currency():
    result = get_currency.invoke({"from_c": "USD", "to_c": "JPY", "amount": 1000})
    assert "result" in result, f"FAIL: {result}"
    log.info("PASS | test_currency | USD->JPY conversion done")
```

### Git Commit
```bash
git add .
git commit -m "feat: add weather/flight/attractions/currency/timezone tools"
```

---

## 🔁 Phase 3 — LangChain Middleware

### AI Agent Prompt for Phase 3

```
You are a senior LangChain developer. Build a 3-part middleware pipeline.

Context: This is a travel voice assistant. All user input is transcribed text from voice.
All output will be converted back to speech.

--- MIDDLEWARE 1: Pre-Model (middleware/pre_model.py) ---
Class: PreModelMiddleware
Method: process(user_input: str, conversation_history: list) -> dict

Must do:
1. Strip extra whitespace and fix obvious transcription artifacts
2. Detect if input contains a city name (basic keyword check)
3. Inject last 3 turns of conversation_history into context
4. Add metadata: { timestamp, input_length, detected_city or None }
5. Return: { cleaned_input, context, metadata }

--- MIDDLEWARE 2: Dynamic Prompt Builder (middleware/dynamic_prompt.py) ---
Class: DynamicPromptBuilder
Method: build(cleaned_input: str, context: dict, agent_type: str) -> str

Must do:
1. Load base system prompt for travel assistant
2. Based on agent_type (weather/flight/attractions/currency/timezone/general),
   inject a specialized instruction block
3. Append recent conversation context
4. Return final assembled prompt string

Base system prompt:
"You are a friendly, concise travel planning assistant. The user is talking to you
via voice, so keep responses under 3 sentences unless asked for more detail.
Always confirm location before fetching data. Be warm and helpful."

Specialized blocks per agent_type:
- weather:     "Focus on temperature, what to pack, and best time to visit."
- flight:      "Mention price, stops, and travel duration. Suggest booking early."
- attractions: "List top 3 only. Include one fun fact per place."
- currency:    "Give the converted amount and a local purchasing power example."
- timezone:    "State the time difference from the user's likely location (assume US EST)."
- general:     "Answer conversationally. Offer to fetch specific data if relevant."

--- MIDDLEWARE 3: Post-Model (middleware/post_model.py) ---
Class: PostModelMiddleware
Method: process(raw_response: str) -> str

Must do:
1. Remove any markdown formatting (**, ##, -, *) — voice cannot speak markdown
2. Remove URLs if any
3. Truncate to max 400 characters if too long (add "...for more details, just ask")
4. Return clean plain text ready for TTS
```

### Test Case — `tests/test_phase3.py`
```python
"""Phase 3 Test: Middleware Pipeline"""
import pytest
import logging

log = logging.getLogger(__name__)

from middleware.pre_model import PreModelMiddleware
from middleware.dynamic_prompt import DynamicPromptBuilder
from middleware.post_model import PostModelMiddleware

def test_pre_model_cleans_input():
    mw = PreModelMiddleware()
    result = mw.process("  what is the weather in   Tokyo  ", [])
    assert result["cleaned_input"] == "what is the weather in Tokyo"
    assert result["metadata"]["detected_city"] == "Tokyo"
    log.info("PASS | test_pre_model_cleans_input")

def test_dynamic_prompt_weather():
    builder = DynamicPromptBuilder()
    prompt = builder.build("weather in Paris", {}, "weather")
    assert "pack" in prompt.lower()
    log.info("PASS | test_dynamic_prompt_weather")

def test_post_model_removes_markdown():
    mw = PostModelMiddleware()
    raw = "**Tokyo** is great! Visit ##Shibuya and [click here](http://x.com)"
    result = mw.process(raw)
    assert "**" not in result
    assert "http" not in result
    log.info("PASS | test_post_model_removes_markdown")

def test_post_model_truncates():
    mw = PostModelMiddleware()
    long_text = "a" * 500
    result = mw.process(long_text)
    assert len(result) <= 430
    log.info("PASS | test_post_model_truncates")
```

### Git Commit
```bash
git add .
git commit -m "feat: add langchain middleware pipeline pre and post model"
```

---

## 🔄 Phase 4 — Model Layer (Ollama + Groq + Retry/Fallback)

### AI Agent Prompt for Phase 4

```
You are a senior Python/LangChain developer. Build a model invocation layer
with retry and fallback support.

File: agents/model_layer.py

Requirements:

1. PRIMARY MODEL: Ollama (local LLM)
   - Use langchain_community.llms.Ollama
   - Model: from config (default: llama3)
   - Base URL: from config

2. FALLBACK MODEL: Groq API
   - Use langchain_groq.ChatGroq
   - Model: mixtral-8x7b-32768
   - API key from config

3. RETRY LOGIC:
   - Wrap Ollama call in a retry loop: max 3 attempts, 2 second delay
   - If all 3 Ollama attempts fail → automatically switch to Groq
   - If Groq also fails → return {"error": "All models unavailable"}

4. Class: ModelLayer
   Method: invoke(prompt: str) -> str

   Flow:
   a. Try Ollama up to 3 times with 2s delay between retries
   b. Log each retry attempt to logs/test_results.log
   c. On Ollama success → return response text
   d. On Ollama failure after 3 retries → log "Falling back to Groq"
   e. Try Groq once
   f. On Groq success → return response text
   g. On Groq failure → return error string

Use Python's time.sleep for delays.
Log all events using Python logging to logs/test_results.log
```

### Test Case — `tests/test_phase4.py`
```python
"""Phase 4 Test: Model Retry and Fallback"""
import pytest
import logging
from unittest.mock import patch, MagicMock

log = logging.getLogger(__name__)

from agents.model_layer import ModelLayer

def test_ollama_success():
    with patch("agents.model_layer.Ollama") as mock_ollama:
        mock_ollama.return_value.invoke.return_value = "Tokyo is beautiful"
        layer = ModelLayer()
        result = layer.invoke("Tell me about Tokyo")
        assert "Tokyo" in result
        log.info("PASS | test_ollama_success")

def test_ollama_retry_then_groq_fallback():
    with patch("agents.model_layer.Ollama") as mock_ollama, \
         patch("agents.model_layer.ChatGroq") as mock_groq:
        mock_ollama.return_value.invoke.side_effect = Exception("Ollama down")
        mock_groq.return_value.invoke.return_value = MagicMock(content="Groq response")
        layer = ModelLayer()
        result = layer.invoke("What is the weather?")
        assert result == "Groq response"
        log.info("PASS | test_ollama_retry_then_groq_fallback")

def test_all_models_fail():
    with patch("agents.model_layer.Ollama") as mock_ollama, \
         patch("agents.model_layer.ChatGroq") as mock_groq:
        mock_ollama.return_value.invoke.side_effect = Exception("Ollama down")
        mock_groq.return_value.invoke.side_effect = Exception("Groq down")
        layer = ModelLayer()
        result = layer.invoke("Anything")
        assert "error" in result.lower() or "unavailable" in result.lower()
        log.info("PASS | test_all_models_fail")
```

### Git Commit
```bash
git add .
git commit -m "feat: add model retry and fallback mechanism ollama to groq"
```

---

## 🎙️ Phase 5 — Voice Layer (STT + TTS with Fallback)

### AI Agent Prompt for Phase 5

```
You are a Python developer building a voice layer for a travel assistant.
Build two files: voice/stt.py and voice/tts.py

--- FILE 1: voice/stt.py ---
Class: STTProvider
Method: transcribe(audio_bytes: bytes) -> str

PRIMARY: Sarvam AI STT
  API: POST https://api.sarvam.ai/speech-to-text
  Headers: { "api-subscription-key": SARVAM_API_KEY }
  Body: multipart form with audio file, language_code="en-IN", model="saarika:v2"
  Parse: response.json()["transcript"]

FALLBACK: Deepgram STT (activate if Sarvam raises any Exception)
  API: POST https://api.deepgram.com/v1/listen
  Headers: { "Authorization": f"Token {DEEPGRAM_API_KEY}" }
  Params: { punctuate: true, language: "en" }
  Body: raw audio bytes
  Parse: response.json()["results"]["channels"][0]["alternatives"][0]["transcript"]

If both fail: return ""
Log every attempt, success, and fallback event to logs/test_results.log

--- FILE 2: voice/tts.py ---
Class: TTSProvider
Method: synthesize(text: str) -> bytes  (returns audio bytes, WAV/MP3)

PRIMARY: Sarvam AI TTS
  API: POST https://api.sarvam.ai/text-to-speech
  Headers: { "api-subscription-key": SARVAM_API_KEY }
  Body: JSON { "inputs": [text], "target_language_code": "en-IN",
               "speaker": "meera", "model": "bulbul:v1" }
  Parse: base64 decode response.json()["audios"][0]

FALLBACK: Deepgram TTS (activate if Sarvam raises any Exception)
  API: POST https://api.deepgram.com/v1/speak
  Params: { model: "aura-asteria-en" }
  Headers: { "Authorization": f"Token {DEEPGRAM_API_KEY}",
             "Content-Type": "application/json" }
  Body: JSON { "text": text }
  Returns: response.content (raw audio bytes)

If both fail: return b""
Log everything to logs/test_results.log
```

### Test Case — `tests/test_phase5.py`
```python
"""Phase 5 Test: Voice STT and TTS with Fallback"""
import pytest
import logging
from unittest.mock import patch, MagicMock

log = logging.getLogger(__name__)

from voice.stt import STTProvider
from voice.tts import TTSProvider

DUMMY_AUDIO = b"RIFF....fakeaudiobytes"

def test_stt_sarvam_success():
    with patch("voice.stt.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"transcript": "what is the weather in Tokyo"}
        mock_post.return_value.status_code = 200
        stt = STTProvider()
        result = stt.transcribe(DUMMY_AUDIO)
        assert "Tokyo" in result
        log.info("PASS | test_stt_sarvam_success")

def test_stt_sarvam_fail_deepgram_fallback():
    with patch("voice.stt.requests.post") as mock_post:
        mock_post.side_effect = [
            Exception("Sarvam down"),
            MagicMock(status_code=200, json=lambda: {
                "results": {"channels": [{"alternatives": [{"transcript": "deepgram result"}]}]}
            })
        ]
        stt = STTProvider()
        result = stt.transcribe(DUMMY_AUDIO)
        assert result == "deepgram result"
        log.info("PASS | test_stt_sarvam_fail_deepgram_fallback")

def test_tts_sarvam_success():
    import base64
    fake_audio = base64.b64encode(b"fakeaudio").decode()
    with patch("voice.tts.requests.post") as mock_post:
        mock_post.return_value.json.return_value = {"audios": [fake_audio]}
        mock_post.return_value.status_code = 200
        tts = TTSProvider()
        result = tts.synthesize("Hello Tokyo")
        assert isinstance(result, bytes)
        log.info("PASS | test_tts_sarvam_success")

def test_tts_sarvam_fail_deepgram_fallback():
    with patch("voice.tts.requests.post") as mock_post:
        mock_post.side_effect = [
            Exception("Sarvam TTS down"),
            MagicMock(status_code=200, content=b"deepgram_audio_bytes")
        ]
        tts = TTSProvider()
        result = tts.synthesize("Hello world")
        assert result == b"deepgram_audio_bytes"
        log.info("PASS | test_tts_sarvam_fail_deepgram_fallback")
```

### Git Commit
```bash
git add .
git commit -m "feat: add stt tts voice layer with sarvam and deepgram fallback"
```

---

## 🤖 Phase 6 — Sub-Agents (5 Agents)

### AI Agent Prompt for Phase 6

```
You are building 5 LangChain ReAct sub-agents for a travel assistant system.
Each agent has ONE specific responsibility. Each agent file goes in agents/ folder.

Use: from langchain.agents import create_react_agent, AgentExecutor

For each agent:
- Accept a query: str parameter
- Use the appropriate tool from tools/ folder
- Return structured plain text response (no markdown)
- Handle tool errors gracefully

--- AGENT 1: agents/weather_agent.py ---
Name: WeatherAgent
Tool: get_weather (from tools/weather_tool.py)
System role: "You are a weather specialist. Extract the city from the user's
  question, call get_weather, and respond with temperature, conditions,
  and 1-2 packing tips. Keep it under 2 sentences."

--- AGENT 2: agents/flight_agent.py ---
Name: FlightAgent
Tool: get_flights (from tools/flight_tool.py)
System role: "You are a flight search specialist. Extract the departure city,
  destination city, and travel date from the query. Call get_flights and
  return the cheapest option with price, stops, and airline. One sentence."

--- AGENT 3: agents/attractions_agent.py ---
Name: AttractionsAgent
Tool: get_attractions (from tools/attractions_tool.py)
System role: "You are a tourism expert. Extract the city, call get_attractions,
  and mention the top 3 places with a brief one-word description each."

--- AGENT 4: agents/currency_agent.py ---
Name: CurrencyAgent
Tool: get_currency (from tools/currency_tool.py)
System role: "You are a currency conversion specialist. Extract from_currency,
  to_currency, and amount. Call get_currency and state the converted amount.
  Add one practical context sentence like 'that covers a nice dinner'."

--- AGENT 5: agents/timezone_agent.py ---
Name: TimezoneAgent
Tool: get_timezone (from tools/timezone_tool.py)
System role: "You are a timezone expert. Extract the city or timezone from the
  query, call get_timezone, and report the current local time and difference
  from US Eastern Time."

Each agent class must have:
  def run(self, query: str) -> str
```

### Git Commit
```bash
git add .
git commit -m "feat: add sub-agents weather flight attractions currency timezone"
```

---

## 🧠 Phase 7 — Supervisor Agent + LangGraph

### AI Agent Prompt for Phase 7

```
You are building a LangGraph-based supervisor agent system for a travel assistant.

File: graph/travel_graph.py
File: agents/supervisor_agent.py

--- LANGGRAPH STATE ---
from typing import TypedDict

class TravelState(TypedDict):
    user_input: str
    conversation_history: list
    detected_intent: str       # weather/flight/attractions/currency/timezone/general
    sub_agent_response: str
    final_response: str
    error: str

--- SUPERVISOR AGENT (agents/supervisor_agent.py) ---
Class: SupervisorAgent
Method: route(state: TravelState) -> str

The supervisor reads user_input and calls an LLM with this exact prompt:

"You are a routing agent for a travel assistant. Given the user's message,
classify the intent into exactly ONE of these categories:
- weather       (user asks about temperature, forecast, climate, rain, packing)
- flight        (user asks about flights, tickets, travel dates, airports, prices)
- attractions   (user asks what to visit, tourist spots, things to do, places)
- currency      (user asks about money, exchange rate, how much in X currency)
- timezone      (user asks about time, what time is it, time difference)
- general       (anything else, greetings, general travel questions)

User message: {user_input}
Respond with ONLY the category word, nothing else."

Return the category string (strip whitespace, lowercase).

--- LANGGRAPH GRAPH (graph/travel_graph.py) ---
Build a StateGraph with these nodes:

Nodes:
1. pre_middleware     → runs PreModelMiddleware.process()
2. supervisor         → runs SupervisorAgent.route() → sets detected_intent
3. weather_node       → runs WeatherAgent.run() if intent==weather
4. flight_node        → runs FlightAgent.run() if intent==flight
5. attractions_node   → runs AttractionsAgent.run() if intent==attractions
6. currency_node      → runs CurrencyAgent.run() if intent==currency
7. timezone_node      → runs TimezoneAgent.run() if intent==timezone
8. general_node       → runs ModelLayer.invoke() with general prompt
9. post_middleware    → runs PostModelMiddleware.process() on sub_agent_response

Edges:
START → pre_middleware → supervisor → conditional_edge based on detected_intent
All sub-agent nodes → post_middleware → END

Expose:
  def run_graph(user_input: str, history: list) -> str
```

### Git Commit
```bash
git add .
git commit -m "feat: add supervisor agent and langgraph travel state machine"
```

---

## 🔗 Phase 8 — Full Integration (Voice → Graph → Voice)

### AI Agent Prompt for Phase 8

```
You are building the full integration FastAPI server for a travel voice assistant.

File: server.py

Build a FastAPI server with these endpoints:

1. POST /voice/query
   - Accepts: multipart form with audio_file (bytes)
   - Step 1: Call STTProvider().transcribe(audio_bytes) → user_text
   - Step 2: Load conversation_history from in-memory dict keyed by session_id header
   - Step 3: Call run_graph(user_text, conversation_history) → response_text
   - Step 4: Append both user_text and response_text to history (keep last 10 turns)
   - Step 5: Call TTSProvider().synthesize(response_text) → audio_bytes
   - Step 6: Return audio_bytes as StreamingResponse with media_type="audio/wav"
   - Log every step with timestamp to logs/test_results.log

2. POST /text/query
   - Accepts: JSON { "text": str, "session_id": str }
   - Skips STT and TTS, runs graph directly
   - Returns: JSON { "response": str }
   - Useful for testing without audio hardware

3. GET /health
   - Returns: { "status": "ok", "ollama": true/false, "sarvam": true/false }
   - Check Ollama by pinging OLLAMA_BASE_URL/api/tags
   - Check Sarvam by checking if SARVAM_API_KEY is set

In-memory session store:
sessions: dict[str, list] = {}   # session_id -> conversation_history list

Run command: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Git Commit
```bash
git add .
git commit -m "feat: integrate voice pipeline with langgraph multi-agent system"
```

---

## 🧪 Phase 9 — All Tests + Server

### Master Test Runner — `tests/run_all_tests.py`
```python
"""Run all phase tests and log pass/fail to logs/test_results.log"""
import subprocess
import logging
import datetime

logging.basicConfig(
    filename="logs/test_results.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

phases = [
    "tests/test_phase2.py",
    "tests/test_phase3.py",
    "tests/test_phase4.py",
    "tests/test_phase5.py",
]

log.info("=" * 60)
log.info(f"MASTER TEST RUN STARTED: {datetime.datetime.now()}")
log.info("=" * 60)

all_passed = True

for phase_file in phases:
    result = subprocess.run(
        ["pytest", phase_file, "-v", "--tb=short"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        log.info(f"PASS | {phase_file} | All tests passed")
    else:
        log.error(f"FAIL | {phase_file}\n{result.stdout}\n{result.stderr}")
        all_passed = False

if all_passed:
    log.info("ALL PHASES PASSED")
    print("All tests passed. Check logs/test_results.log")
else:
    log.error("SOME PHASES FAILED")
    print("Some tests failed. Check logs/test_results.log")
```

### Integration Test — `tests/test_integration.py`
```python
"""Full integration test via /text/query endpoint (server must be running)"""
import requests
import logging

log = logging.getLogger(__name__)
BASE = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200
    log.info("PASS | test_health")

def test_weather_query():
    r = requests.post(f"{BASE}/text/query",
                      json={"text": "What is the weather in Paris?", "session_id": "test1"})
    assert r.status_code == 200
    assert "response" in r.json()
    log.info("PASS | test_weather_query")

def test_currency_query():
    r = requests.post(f"{BASE}/text/query",
                      json={"text": "Convert 500 dollars to euros", "session_id": "test2"})
    assert r.status_code == 200
    log.info("PASS | test_currency_query")

def test_context_retention():
    sid = "test_context"
    requests.post(f"{BASE}/text/query",
                  json={"text": "I want to visit Tokyo", "session_id": sid})
    r = requests.post(f"{BASE}/text/query",
                      json={"text": "What time is it there?", "session_id": sid})
    resp = r.json().get("response", "").lower()
    assert "tokyo" in resp or "japan" in resp or "jst" in resp
    log.info("PASS | test_context_retention")
```

### Git Commit
```bash
git add .
git commit -m "feat: add fastapi server and run all phase tests with pass log"
```

---

## 🧹 Phase 10 — Final Cleanup + Tag Release

```bash
git add .
git commit -m "fix: resolve integration issues and finalize all test cases"

# Tag the final release
git tag -a v1.0.0 -m "Travel Voice Assistant v1.0 - All phases complete"
git push origin main --tags
```

---

## 📋 Expected Git Log at End of Day 2

```
* fix: resolve integration issues and finalize all test cases
* feat: add fastapi server and run all phase tests with pass log
* feat: integrate voice pipeline with langgraph multi-agent system
* feat: add supervisor agent and langgraph travel state machine
* feat: add sub-agents weather flight attractions currency timezone
* feat: add stt tts voice layer with sarvam and deepgram fallback
* feat: add model retry and fallback mechanism ollama to groq
* feat: add langchain middleware pipeline pre and post model
* feat: add weather/flight/attractions/currency/timezone tools
* feat: initial project scaffold and config setup
```

---

## ⚡ Quick Start Commands

```bash
# 1. Clone and setup
git clone <your-repo>
cd travel-voice-assistant
pip install -r requirements.txt

# 2. Fill env
cp .env.example .env
# Edit .env with your API keys

# 3. Start Ollama locally
ollama pull llama3
ollama serve

# 4. Start server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# 5. Run all unit tests
python tests/run_all_tests.py

# 6. Run integration tests (server must be up)
pytest tests/test_integration.py -v

# 7. View logs
cat logs/test_results.log
```

---

## 🔑 Technology Stack Summary

| Component | Primary | Fallback |
|-----------|---------|----------|
| STT (Speech to Text) | Sarvam AI | Deepgram |
| TTS (Text to Speech) | Sarvam AI | Deepgram |
| LLM | Ollama local (llama3) | Groq API |
| Orchestration | LangGraph StateGraph | — |
| Agent Framework | LangChain ReAct | — |
| Middleware | Custom Pre/Post + Dynamic Prompt | — |
| Agent Pattern | Supervisor routes to Sub-agents | — |
| Server | FastAPI + Uvicorn | — |
| Testing | Pytest + unittest.mock | — |
| Logging | Python logging to logs/test_results.log | — |