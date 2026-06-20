# Fetches the current local time for a given timezone using TimeAPI.io.

import requests
from langchain_core.tools import tool


@tool
def get_timezone(timezone: str) -> dict:
    """Get the current date and time for a given IANA timezone.

    Args:
        timezone: A valid IANA timezone string (e.g. 'Asia/Tokyo', 'Europe/London').

    Useful for telling the user what time it is at their destination and
    calculating the offset from their home timezone.
    """
    url = "https://timeapi.io/api/Time/current/zone"
    params = {"timeZone": timezone}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "timezone": data.get("timeZone"),
            "datetime": data.get("dateTime"),
            "hour": data.get("hour"),
            "minute": data.get("minute"),
        }
    except requests.exceptions.HTTPError:
        return {"error": f"Timezone API returned {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
