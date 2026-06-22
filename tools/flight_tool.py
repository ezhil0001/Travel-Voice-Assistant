# Handles flight search via FlightAPI.io — a single REST call using the API key
# embedded directly in the URL path. Replaced the previous Amadeus OAuth2 setup
# because the Amadeus self-service portal no longer accepts new registrations.

import logging
import time
import requests
from langchain_core.tools import tool
from config import settings

logger = logging.getLogger(__name__)

# The upstream API returns HTTP 400 intermittently (Skyscanner-backed service
# under load).  Retrying a handful of times resolves it in practice.
_MAX_RETRIES   = 5
_RETRY_DELAY_S = 1.5


def _parse_flight_response(data: dict, date: str) -> list:
    """Extract flight offers from the FlightAPI.io onewaytrip response.

    The actual response shape (confirmed via live API inspection) uses:
      - data["itineraries"]  — list of itinerary objects, each with:
          leg_ids              list[str]
          pricing_options      list[{price: {amount: int}}]
      - data["legs"]          — dict keyed by leg-id, each with:
          departure            ISO-8601 datetime string
          arrival              ISO-8601 datetime string
          stop_count           int
          marketing_carrier_ids list[int]
      - data["carriers"]      — list of {id: int, name: str, display_code: str}

    We extract up to 3 results and look up carrier names from the carriers list.
    """
    itineraries = data.get("itineraries") or []
    legs_raw     = data.get("legs") or []
    carriers_raw = data.get("carriers") or []

    # legs comes as a list — build an id → leg dict for O(1) lookup
    legs_map: dict = {}
    if isinstance(legs_raw, list):
        legs_map = {leg["id"]: leg for leg in legs_raw if "id" in leg}
    elif isinstance(legs_raw, dict):
        legs_map = legs_raw

    # Build carrier lookup: id → display_code (IATA) or name
    carrier_lookup: dict = {}
    for c in carriers_raw:
        cid = c.get("id")
        if cid is not None:
            carrier_lookup[cid] = c.get("display_code") or c.get("name", "Unknown")

    results = []
    for it in itineraries[:3]:
        # Price from first pricing option
        options = it.get("pricing_options") or []
        price = "N/A"
        if options:
            price_obj = options[0].get("price") or {}
            raw = price_obj.get("amount")
            if raw is not None:
                price = str(raw)

        # Leg details
        leg_id = (it.get("leg_ids") or [None])[0]
        leg    = legs_map.get(leg_id, {}) if isinstance(legs_map, dict) else {}

        departure  = leg.get("departure", date)
        arrival    = leg.get("arrival", date)
        stop_count = leg.get("stop_count", 0)

        # Carrier — first marketing carrier id
        carrier_ids = leg.get("marketing_carrier_ids") or []
        airline = carrier_lookup.get(carrier_ids[0], "Unknown") if carrier_ids else "Unknown"

        results.append({
            "airline":   airline,
            "price":     price,
            "currency":  "USD",
            "stops":     stop_count,
            "departure": departure,
            "arrival":   arrival,
        })

    return results


@tool
def get_flights(origin: str, destination: str, date: str) -> list:
    """Search for available flights between two airports on a given date.

    Requires 3-letter IATA airport codes — the LLM must resolve city names
    to IATA codes before calling this tool (e.g. New York → JFK, Tokyo → NRT).

    Args:
        origin:      Uppercase 3-letter IATA code of the departure airport (e.g. 'MAA').
        destination: Uppercase 3-letter IATA code of the arrival airport (e.g. 'BKK').
        date:        Departure date in YYYY-MM-DD format.

    Returns up to 3 offers each containing airline, price, stops, departure, and arrival.
    """
    if not settings.FLIGHTAPI_KEY:
        return [{"error": "FLIGHTAPI_KEY is not configured"}]

    url = (
        f"https://api.flightapi.io/onewaytrip/"
        f"{settings.FLIGHTAPI_KEY}/{origin}/{destination}/{date}/1/0/0/Economy/USD"
    )
    logger.info("get_flights | url=%s", url)

    last_error = "Unknown error"
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=20)
            logger.info(
                "get_flights | attempt=%d | status=%d | body=%.300s",
                attempt, response.status_code, response.text,
            )

            if response.status_code == 200:
                data    = response.json()
                results = _parse_flight_response(data, date)

                if not results:
                    logger.warning(
                        "get_flights | no results parsed | top-level keys=%s",
                        list(data.keys()),
                    )
                    return [{"error": f"No flights found for {origin}→{destination} on {date}"}]

                logger.info("get_flights | found %d offers", len(results))
                return results

            # 4xx other than 400 are not retryable
            if response.status_code != 400:
                response.raise_for_status()

            last_error = f"HTTP {response.status_code}"

        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            logger.error("get_flights | attempt=%d | request failed: %s", attempt, exc)

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY_S)

    logger.error("get_flights | all %d attempts failed | last_error=%s", _MAX_RETRIES, last_error)
    return [{"error": f"Flights API unavailable after {_MAX_RETRIES} attempts ({last_error})"}]
