# Verifies that all required environment variables and config keys are present at startup.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_keys_present():
    """All expected env-var names must be defined in settings.py."""
    from config import settings

    expected_keys = [
        "OPENWEATHER_API_KEY",
        "FLIGHTAPI_KEY",          # replaced Amadeus after self-service portal shutdown
        "GEONAMES_USERNAME",       # replaced OpenTripMap — uses free username-based auth
        "EXCHANGERATE_API_KEY",
        "SARVAM_API_KEY",
        "DEEPGRAM_API_KEY",
        "GROQ_API_KEY",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
    ]
    for key in expected_keys:
        assert hasattr(settings, key), f"Missing config key: {key}"


def test_ollama_defaults():
    """OLLAMA_BASE_URL and OLLAMA_MODEL must have fallback defaults."""
    from config import settings

    assert settings.OLLAMA_BASE_URL is not None, "OLLAMA_BASE_URL should have a default"
    assert settings.OLLAMA_MODEL is not None, "OLLAMA_MODEL should have a default"


def test_folder_structure():
    """All required package folders must exist."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    required_dirs = [
        "agents", "middleware", "voice", "tools", "graph", "config", "tests", "logs",
    ]
    for d in required_dirs:
        path = os.path.join(base, d)
        assert os.path.isdir(path), f"Missing folder: {d}/"


def test_env_file_exists():
    """A .env file must exist at the project root."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.isfile(os.path.join(base, ".env")), ".env file not found"


def test_requirements_file_exists():
    """requirements.txt must exist and contain core packages."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    req_path = os.path.join(base, "requirements.txt")
    assert os.path.isfile(req_path), "requirements.txt not found"

    with open(req_path) as f:
        content = f.read()

    for pkg in ["langchain", "langgraph", "fastapi", "uvicorn", "pytest", "groq"]:
        assert pkg in content, f"Package '{pkg}' missing from requirements.txt"
