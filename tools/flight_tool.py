# Handles flight search via FlightAPI.io — a single REST call using the API key
# embedded directly in the URL path. Replaced the previous Amadeus OAuth2 setup
# because the Amadeus self-service portal no longer accepts new registrations.

import requests
from langchain_core.tools import tool
from config import settings


@tool
def get_flights(origin: str, destination: str, date: str) -> list:
    """Search for available flights between two airports on a given date.

    Requires 3-letter IATA airport codes — the LLM must resolve city names
    to IATA codes before calling this tool (e.g. New York → JFK, Tokyo → NRT).

    Args:
        origin: Uppercase 3-letter IATA code of the departure airport (e.g. 'JFK').
        destination: Uppercase 3-letter IATA code of the arrival airport (e.g. 'NRT').
        date: Departure date in YYYY-MM-DD format.

    Returns up to 3 offers each containing airline, price, stops, departure, and arrival.
    """
    if not settings.FLIGHTAPI_KEY:
        return [{"error": "FLIGHTAPI_KEY is not configured"}]

    # API key is injected directly into the URL path — no token exchange needed
    url = (
        f"https://api.flightapi.io/onewaytrip/"
        f"{settings.FLIGHTAPI_KEY}/{origin}/{destination}/{date}/1/0/0/Economy/USD"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        legs = data.get("legs", [])
        if not legs:
            return [{"error": "No flights found for these parameters"}]

        results = []
        # Cap at 3 results to keep the voice response concise
        for leg in legs[:3]:
            segments = leg.get("segments", [])
            results.append({
                "airline": leg.get("airlineCode", "Unknown"),
                "price": leg.get("price", {}).get("total", "N/A"),
                "currency": "USD",
                "stops": max(len(segments) - 1, 0),
                "departure": leg.get("departureTime", date),
                "arrival": leg.get("arrivalTime", date),
            })

        return results

    except requests.exceptions.HTTPError:
        return [{"error": f"Flights API returned {response.status_code}: {response.text}"}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"Request failed: {str(e)}"}]
