# Tests each external API tool in isolation to catch integration errors early.
#
# HTTP calls are mocked so these tests validate tool logic and response shaping
# without requiring real API keys. A separate integration test suite (not committed)
# is used against live endpoints during manual QA.

import logging
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    filename="logs/test_results.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

from tools.weather_tool import get_weather
from tools.flight_tool import get_flights
from tools.attractions_tool import get_attractions
from tools.timezone_tool import get_timezone
from tools.currency_tool import get_currency


def _mock_response(json_data, status_code=200):
    """Build a minimal mock that mimics requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    return mock


def test_weather():
    fake_payload = {
        "name": "Tokyo",
        "main": {"temp": 18.5, "feels_like": 17.2, "humidity": 60},
        "weather": [{"description": "clear sky"}],
        "wind": {"speed": 3.5},
    }
    with patch("tools.weather_tool.requests.get", return_value=_mock_response(fake_payload)):
        result = get_weather.invoke({"city": "Tokyo"})

    assert "temperature" in result, f"Weather tool failed: {result}"
    assert result["city"] == "Tokyo"
    assert result["temperature"] == 18.5
    log.info("PASS | test_weather | response shaped correctly")


def test_flights():
    # FlightAPI.io returns a `legs` list — no OAuth2 token call needed
    offers_payload = {
        "legs": [
            {
                "airlineCode": "JL",
                "price": {"total": "750.00"},
                "segments": [{"dep": "JFK"}, {"arr": "NRT"}],
                "departureTime": "2025-03-15T10:00:00",
                "arrivalTime": "2025-03-16T14:30:00",
            }
        ]
    }

    with patch("tools.flight_tool.requests.get", return_value=_mock_response(offers_payload)):
        result = get_flights.invoke({"origin": "JFK", "destination": "NRT", "date": "2025-03-15"})

    assert isinstance(result, list), f"Flights tool failed: {result}"
    assert result[0]["airline"] == "JL"
    assert result[0]["stops"] == 1   # 2 segments → 1 stop
    log.info("PASS | test_flights | FlightAPI.io offers parsed correctly")


def test_attractions():
    geoname_payload = {"lat": "48.8566", "lon": "2.3522"}
    radius_payload = {
        "features": [
            {"properties": {"name": "Eiffel Tower", "kinds": "architecture", "dist": 1200}},
            {"properties": {"name": "Louvre Museum", "kinds": "museums", "dist": 2100}},
        ]
    }

    with patch("tools.attractions_tool.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response(geoname_payload),
            _mock_response(radius_payload),
        ]
        result = get_attractions.invoke({"city": "Paris"})

    assert isinstance(result, list), f"Attractions tool failed: {result}"
    assert result[0]["name"] == "Eiffel Tower"
    log.info("PASS | test_attractions | attraction list returned correctly")


def test_timezone():
    fake_payload = {
        "timeZone": "Asia/Tokyo",
        "dateTime": "2025-03-15T14:30:00",
        "hour": 14,
        "minute": 30,
    }
    with patch("tools.timezone_tool.requests.get", return_value=_mock_response(fake_payload)):
        result = get_timezone.invoke({"timezone": "Asia/Tokyo"})

    assert "datetime" in result, f"Timezone tool failed: {result}"
    assert result["hour"] == 14
    log.info("PASS | test_timezone | timezone response shaped correctly")


def test_currency():
    fake_payload = {
        "result": "success",
        "conversion_result": 150230.0,
        "conversion_rate": 150.23,
    }
    with patch("tools.currency_tool.requests.get", return_value=_mock_response(fake_payload)):
        result = get_currency.invoke({"from_c": "USD", "to_c": "JPY", "amount": 1000})

    assert "result" in result, f"Currency tool failed: {result}"
    assert result["result"] == 150230.0
    assert result["rate"] == 150.23
    log.info("PASS | test_currency | currency conversion shaped correctly")
