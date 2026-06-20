# Queries OpenTripMap to return a list of tourist attractions near a given location.

import requests
from langchain_core.tools import tool
from config import settings


def _geocode_city(city: str) -> tuple[float, float] | None:
    """Resolve a city name to (latitude, longitude) via OpenTripMap's geoname endpoint.

    This is a prerequisite step because the radius search needs coordinates,
    not a city name string.
    """
    url = "https://api.opentripmap.com/0.1/en/places/geoname"
    params = {"name": city, "apikey": settings.OPENTRIPMAP_API_KEY}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data["lat"]), float(data["lon"])
    except (requests.exceptions.RequestException, KeyError):
        return None


@tool
def get_attractions(city: str) -> list:
    """Find top tourist attractions near a city using OpenTripMap.

    Args:
        city: The city name to search around (e.g. 'Paris').

    Returns a list of places with name, category tags, and distance in metres.
    """
    if not settings.OPENTRIPMAP_API_KEY:
        return [{"error": "OPENTRIPMAP_API_KEY is not configured"}]

    coords = _geocode_city(city)
    if coords is None:
        return [{"error": f"Could not geocode city: {city}"}]

    lat, lon = coords
    url = "https://api.opentripmap.com/0.1/en/places/radius"
    params = {
        "radius": 5000,
        "lon": lon,
        "lat": lat,
        "limit": 10,
        "apikey": settings.OPENTRIPMAP_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        features = response.json().get("features", [])

        results = []
        for feature in features:
            props = feature.get("properties", {})
            name = props.get("name", "").strip()
            if not name:
                continue
            results.append({
                "name": name,
                "kinds": props.get("kinds", ""),
                "distance": props.get("dist", 0),
            })

        return results if results else [{"error": "No attractions found near this city"}]

    except requests.exceptions.HTTPError:
        return [{"error": f"Attractions API returned {response.status_code}: {response.text}"}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"Request failed: {str(e)}"}]
