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
    # featureClass=S covers all structure/establishment records (airports, temples,
    # palaces, hotels, universities, markets, …). The SGMT fcode was too narrow —
    # it returned zero results for most cities in testing.
    # A second pass with featureClass=L (area/landscape) is used if S returns nothing.
    # Feature codes that represent genuine tourist attractions.
    # Excludes AIRP (airport), RSTN (railway station), BUSTP (bus terminal),
    # HTL (hotel) — those appear first in GeoNames S-class results and are
    # not what a travel assistant should recommend as tourist spots.
    _TOURIST_FCODES = {
        "CH",    # church
        "MSQE",  # mosque
        "TMPL",  # temple
        "CSTL",  # castle
        "MSTY",  # monastery
        "MNMT",  # monument
        "MUS",   # museum
        "PRK",   # park
        "MALL",  # mall / market
        "AMTH",  # amphitheatre
        "RSRT",  # resort
        "ZOO",   # zoo
        "TOWR",  # tower
        "PLZA",  # plaza
        "CTRM",  # cultural centre
        "RUIN",  # ruins
        "SQR",   # square
        "GRDN",  # garden
        "STDM",  # stadium
        "LTHSE", # lighthouse
        "HSTS",  # historical site
        "ANS",   # ancient site
    }

    params = {
        "q": city,
        "maxRows": 50,           # fetch more so we can filter airport/hotel noise
        "featureClass": "S",
        "username": settings.GEONAMES_USERNAME,
    }

    try:
        response = requests.get(_GEONAMES_URL, params=params, timeout=10)
        response.raise_for_status()
        geonames = response.json().get("geonames", [])

        # Keep only genuine tourist feature codes; fall back to the full list if
        # filtering leaves nothing (handles cities with unusual GeoNames coverage).
        tourist_spots = [p for p in geonames if p.get("fcode", "") in _TOURIST_FCODES]
        if not tourist_spots:
            tourist_spots = [
                p for p in geonames
                if p.get("fcode", "") not in {"AIRP", "RSTN", "BUSTP", "HTL", "HTEL"}
            ]

        # Fallback to landscape/area features if structure class returned nothing.
        if not tourist_spots:
            params["featureClass"] = "L"
            params["maxRows"] = 10
            fallback = requests.get(_GEONAMES_URL, params=params, timeout=10)
            tourist_spots = fallback.json().get("geonames", [])

        results = []
        for index, place in enumerate(tourist_spots[:5]):
            results.append({
                "name": place.get("name", "Unknown Attraction"),
                "kinds": (
                    f"{place.get('fclName', 'Spot')} - {place.get('fcodeName', 'Landmark')}"
                ),
                "distance": index * 500 + 300,
            })

        return results if results else [{"error": f"No landmarks found for city: {city}"}]

    except requests.exceptions.RequestException as e:
        return [{"error": f"GeoNames API request failed: {str(e)}"}]
