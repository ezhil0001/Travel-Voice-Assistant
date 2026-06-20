# Converts an amount between two currencies using the ExchangeRate-API.

import requests
from langchain_core.tools import tool
from config import settings


@tool
def get_currency(from_c: str, to_c: str, amount: float) -> dict:
    """Convert an amount from one currency to another using live exchange rates.

    Args:
        from_c: Source currency code (e.g. 'USD').
        to_c: Target currency code (e.g. 'JPY').
        amount: The numeric amount to convert.

    Returns the converted amount, exchange rate, and both currency codes.
    """
    if not settings.EXCHANGERATE_API_KEY:
        return {"error": "EXCHANGERATE_API_KEY is not configured"}

    url = (
        f"https://v6.exchangerate-api.com/v6/{settings.EXCHANGERATE_API_KEY}"
        f"/pair/{from_c.upper()}/{to_c.upper()}/{amount}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("result") != "success":
            return {"error": data.get("error-type", "Unknown error from ExchangeRate-API")}

        return {
            "from_currency": from_c.upper(),
            "to_currency": to_c.upper(),
            "amount": amount,
            "result": data.get("conversion_result"),
            "rate": data.get("conversion_rate"),
        }
    except requests.exceptions.HTTPError:
        return {"error": f"Currency API returned {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
