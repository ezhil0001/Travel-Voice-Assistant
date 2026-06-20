# Wraps the OpenWeatherMap API to fetch current weather conditions for a given city.

import requests
from langchain_core.tools import tool
from config import settings


@tool
def get_weather(city: str) -> dict:
    """Fetch current weather for a city using OpenWeatherMap.

    Returns temperature, feels-like, description, humidity, and wind speed.
    Useful when the user asks about current conditions or what to pack.
    """
    if not settings.OPENWEATHER_API_KEY:
        return {"error": "OPENWEATHER_API_KEY is not configured"}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": settings.OPENWEATHER_API_KEY,
        "units": "metric",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
        }
    except requests.exceptions.HTTPError as e:
        return {"error": f"Weather API returned {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
