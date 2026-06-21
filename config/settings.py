import os
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY   = os.getenv("OPENWEATHER_API_KEY")
# Amadeus self-service portal is decommissioned — replaced with FlightAPI.io
FLIGHTAPI_KEY         = os.getenv("FLIGHTAPI_KEY")
# GeoNames is used for attraction lookups — requires only a free username, no API key
GEONAMES_USERNAME     = os.getenv("GEONAMES_USERNAME", "sabari_07045")
EXCHANGERATE_API_KEY  = os.getenv("EXCHANGERATE_API_KEY")
SARVAM_API_KEY        = os.getenv("SARVAM_API_KEY")
DEEPGRAM_API_KEY      = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY          = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL          = os.getenv("OLLAMA_MODEL", "llama3")

# Middleware tuning — controlled via .env so behaviour can be adjusted without
# touching Python source files (useful for staging vs production configs)
PRE_MODEL_HISTORY_WINDOW  = int(os.getenv("PRE_MODEL_HISTORY_WINDOW", "3"))
POST_MODEL_MAX_CHARS      = int(os.getenv("POST_MODEL_MAX_CHARS", "400"))
POST_MODEL_OVERFLOW_SUFFIX = os.getenv(
    "POST_MODEL_OVERFLOW_SUFFIX", "...for more details, just ask"
)
