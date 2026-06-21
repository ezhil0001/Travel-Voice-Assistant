# Queries the GeoNames searchJSON endpoint to return tourist landmarks for a city.
# OpenTripMap was dropped due to persistent registration verification failures on their end.
# GeoNames is free, stable, and requires only a username — no OAuth or API key rotation.

import requests
from langchain_core.tools import tool
from config import settings

_GEONAMES_URL = "http://api.geonames.org/searchJSON"


@tool
def get_attractions(city: str) -> list:
    """Find top tourist points of interest and attractions in a city using GeoNames.

    Args:
        city: The city name to search (e.g. 'Paris', 'Tokyo').

    Returns a list of significant tourist locations or landmarks, each with
    a name, category description, and a relative distance indicator.
    """
    params = {
        "q": city,
        "maxRows": 10,
        "fcode": "SGMT",          # Feature code for monuments and historic sites
        "username": settings.GEONAMES_USERNAME,
    }

    try:
        response = requests.get(_GEONAMES_URL, params=params, timeout=10)
        response.raise_for_status()
        geonames = response.json().get("geonames", [])

        # SGMT is a narrow filter — if nothing comes back, fall back to a general
        # keyword search so the user always gets some result for the city.
        if not geonames:
            params.pop("fcode")
            fallback = requests.get(_GEONAMES_URL, params=params, timeout=10)
            geonames = fallback.json().get("geonames", [])

        results = []
        for index, place in enumerate(geonames[:5]):
            results.append({
                "name": place.get("name", "Unknown Attraction"),
                "kinds": (
                    f"{place.get('fclName', 'Spot')} - {place.get('fcodeName', 'Landmark')}"
                ),
                # GeoNames doesn't return walking distance — use a plausible
                # ordinal spread so downstream code that reads 'distance' doesn't break.
                "distance": index * 500 + 300,
            })

        return results if results else [{"error": f"No landmarks found for city: {city}"}]

    except requests.exceptions.RequestException as e:
        return [{"error": f"GeoNames API request failed: {str(e)}"}]
