# Travel Voice Assistant using Retell AI

---

## Overview

Configure a conversational voice-powered travel assistant using Retell AI's dashboard and built-in function calling capabilities. The assistant should help users plan trips by providing real-time information about **weather**, **flights**, **attractions**, **currency exchange**, and **time zones** using external APIs configured directly within the Retell agent.

---

## Project Description

Create a voice AI agent entirely within Retell AI platform that can engage in multi-turn conversations about travel planning. Users should be able to call the agent and ask questions naturally, while the assistant fetches real-time data from multiple sources using Retell's function calling feature.

---

## Example User Journey

> **User:** "I'm planning a trip to Tokyo"
>
> **Assistant:** "Tokyo is a great choice! When are you planning to visit?"
>
> **User:** "Maybe next month, around March 15th"
>
> **Assistant:** "Let me check the weather for mid-March in Tokyo... The forecast shows temperatures around 15-18°C with mostly clear skies. It's cherry blossom season - perfect timing! Would you like me to look for flights?"
>
> **User:** "Yes, from New York. Also, what time is it there now?"
>
> **Assistant:** "Currently, it's 2:30 PM in Tokyo, which is 14 hours ahead of New York. I found 3 flights from New York to Tokyo on March 15th. The most affordable option is $750 with one stop."
>
> **User:** "How much is 1000 dollars in Japanese yen?"
>
> **Assistant:** "1000 US dollars equals approximately 150,230 Japanese yen at today's exchange rate."

---

## Configuration Requirements

### 1. Retell AI Agent Setup

Create a new agent in Retell dashboard with:

| Field | Value |
|---|---|
| **Agent Name** | Travel Planning Assistant |
| **Language** | English |
| **First Message** | "Hello! I'm your travel planning assistant. I can help you with weather information, flight searches, tourist attractions, currency conversion, and time zones. Where would you like to explore?" |

> **NOTE:** RETELL ALLOWS YOU TO CREATE VOICE AGENT CONFIGURED AS A **SINGLE PROMPT**, **MULTI PROMPT**, OR A **CONVERSATION FLOW**. YOU ARE ALLOWED TO USE ANY METHOD AS LONG AS THE AGENT HAS ALL OF THE MENTIONED CAPABILITIES.

---

### 2. LLM Configuration

Configure the agent's prompt:

```
You are a helpful and conversational travel planning assistant. You have access to real-time
information about:
- Weather conditions and forecasts
- Flight availability and prices
- Tourist attractions and places to visit
- Currency exchange rates
- Time zones around the world

Guidelines:
1. Be conversational and friendly
2. When a user mentions a destination, proactively offer relevant information
3. Remember context throughout the conversation
4. Provide practical travel advice based on the data
5. If a user asks about weather, also mention what to pack
6. When discussing flights, mention the flight duration if available
7. For currency conversion, provide context (e.g., "that's about the cost of a nice dinner")
8. Always confirm the city/location before making API calls to avoid errors

Keep responses concise but informative. Ask clarifying questions when needed.
```

---

### 3. Function Configuration

Configure the following **FREE APIs** in Retell agent:

---

#### A. Weather Information

- **API:** OpenWeatherMap
- **Endpoint:** `https://api.openweathermap.org/data/2.5/weather`
- **Method:** GET
- **Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `q` | string | City name |
| `appid` | string | Your API key |
| `units` | string | `"metric"` |

---

#### B. Flight Search

- **API:** Amadeus
- **Endpoint:** `https://developers.amadeus.com/self-service/category/flights/api-doc/`

---

#### C. Tourist Attractions

- **API:** OpenTripMap
- **Endpoint:** `https://dev.opentripmap.org/docs#/`

---

#### D. Time Zone Information

- **API:** TimeAPI.io
- **Endpoint:** `https://timeapi.io/api/Time/current/zone`

---

#### E. Currency Exchange

- **API:** ExchangeRate-API
- **Endpoint:** `https://v6.exchangerate-api.com/v6/YOUR_API_KEY/pair/{from}/{to}/{amount}`

---

## Testing Scenarios

### Scenario 1: Basic Weather Query

- **User:** "What's the weather in Paris?"
- **Expected:** Agent calls `get_weather` function and provides current conditions

---

### Scenario 2: Multi-Function Query

- **User:** "I'm going to London next week. What's the weather forecast and what should I visit?"
- **Expected:** Agent calls both `get_weather_forecast` and `get_attractions` functions

---

### Scenario 3: Context Retention

- **User:** "I want to visit Tokyo"
- **Agent:** *(Acknowledges Tokyo)*
- **User:** "What time is it there?"
- **Expected:** Agent remembers Tokyo from previous turn and calls timezone function

---

### Scenario 4: Currency Conversion with Context

- **User:** "How much is $500 in euros?"
- **Expected:** Agent converts and provides practical context

---

### Scenario 5: Complex Travel Planning

- **User:** "I'm planning a trip from New York to Rome in April. What do I need to know?"
- **Expected:** Agent provides weather, timezone difference, currency info, and attractions

---

## Deliverables

### 1. Configured Retell Agent

- Fully functional agent with all APIs integrated
- Proper error handling messages configured
- Natural conversation flow

---

### 2. Documentation (PDF/Doc)

- **Configuration Guide:** Step-by-step setup instructions
- **API Function Details:** List of all configured functions with parameters
- **Prompt Engineering:** Final prompts used and reasoning
- **Testing Results:** Screenshots/transcripts of test conversations

---

### 3. Demo Video (5–7 minutes)

- Show agent configuration in Retell dashboard
- Make 3–4 test calls demonstrating different features
- Show successful API integrations
- Demonstrate error handling