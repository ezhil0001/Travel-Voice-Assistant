import os
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY   = os.getenv("OPENWEATHER_API_KEY")
# Amadeus self-service portal is decommissioned — replaced with FlightAPI.io
FLIGHTAPI_KEY         = os.getenv("FLIGHTAPI_KEY")
OPENTRIPMAP_API_KEY   = os.getenv("OPENTRIPMAP_API_KEY")
EXCHANGERATE_API_KEY  = os.getenv("EXCHANGERATE_API_KEY")
SARVAM_API_KEY        = os.getenv("SARVAM_API_KEY")
DEEPGRAM_API_KEY      = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL          = os.getenv("OLLAMA_MODEL", "llama3")
