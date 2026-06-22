# Flight sub-agent — extends BaseAgent which handles chain construction,
# tool binding, and retry/fallback. This class owns only its tool, its
# system prompt, and its result-formatting logic. No middleware here.

import logging
import re
from datetime import date as _today_date, timedelta
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
from tools.flight_tool import get_flights

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a flight search specialist for a voice travel assistant. "
    "The query contains IATA airport codes (3 uppercase letters) and a date in YYYY-MM-DD format. "
    "Extract origin IATA, destination IATA, and date — then call get_flights. "
    "NEVER pass city names; only use the IATA codes already present in the query. "
    "Return the cheapest option with price, stops, and airline. One sentence. No markdown."
)

_DISPLAY_PROMPT = (
    "You are a flight search specialist. Format the tool result as a clean visual summary using markdown.\n\n"
    "Use this exact layout:\n"
    "**Available Flights — [origin] → [destination]**\n\n"
    "| Airline | Departure | Arrival | Stops | Price |\n"
    "|---------|-----------|---------|-------|-------|\n"
    "| XX | HH:MM | HH:MM | N | $XXX |\n\n"
    "(Include up to 3 rows. Format times as HH:MM from the ISO timestamps. "
    "Highlight the cheapest row with ✅ in front of the airline name.)\n\n"
    "**Tip:** Book early for best fares. Prices shown in USD.\n\n"
    "Use only the data from the tool result. Do not fabricate flight details."
)

_GENERAL_FLIGHT_PROMPT = (
    "You are a knowledgeable travel assistant. The live flight search API is currently unavailable. "
    "Based on your training knowledge, give a helpful answer about the route asked. "
    "Include: typical airlines serving the route, approximate flight duration, number of stops, "
    "and a rough price range in USD. "
    "Be clear this is approximate and recommend checking a booking site for live prices. "
    "No markdown. Two to three sentences maximum."
)


class FlightAgent(BaseAgent):
    """Searches for flights using LangChain tool binding + Runnable chain.

    Extends BaseAgent which builds the bound_chain and format_chain.
    run() strictly handles: tool invocation → LLM formatting → return string.
    Middleware orchestration is handled by ServiceLayer, not this class.
    """

    def __init__(self) -> None:
        super().__init__(tool=get_flights)

    def run(self, query: str, origin: str = "", destination: str = "",
            date: str = "") -> str:
        """Tool invocation → LLM formatting → return plain string.

        The focused query passed by run_agents_node already contains IATA codes
        and a YYYY-MM-DD date (validated by _validate_flight_query). This method
        adds a Python-level guard: if the LLM still produces city names instead
        of IATA codes, the guard overwrites them before the tool is called.
        """
        logger.info("FlightAgent.run | %s → %s on %s", origin, destination, date)
        try:
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]

            response = self._bound_chain.invoke(messages)

            if response.tool_calls:
                args = dict(response.tool_calls[0]["args"])
                logger.info("FlightAgent.run | tool_args=%s", args)

                # Guard: override with Python-extracted values when the LLM
                # returned city names or left fields empty.
                iata_pairs = _extract_iata_and_date(query)
                if iata_pairs["origin"]:
                    args["origin"] = iata_pairs["origin"]
                if iata_pairs["destination"]:
                    args["destination"] = iata_pairs["destination"]
                if iata_pairs["date"]:
                    args["date"] = iata_pairs["date"]

                logger.info("FlightAgent.run | final_args=%s", args)
                tool_result = self._tool.invoke(args)

                if isinstance(tool_result, list) and tool_result and "error" in tool_result[0]:
                    logger.warning("FlightAgent.run | API error: %s — falling back to LLM knowledge", tool_result[0]["error"])
                    return self._llm_fallback(query)

                format_messages = [
                    SystemMessage(content=_DISPLAY_PROMPT),
                    HumanMessage(content=f"Tool result: {tool_result}. Format this as a flight comparison table."),
                ]
                final = self._format_chain.invoke(format_messages)
                return final.content

            return response.content

        except Exception as exc:
            logger.error("FlightAgent.run | failed: %s", exc)
            return self._llm_fallback(query)

    def _llm_fallback(self, query: str) -> str:
        """Use LLM training knowledge when the live flight API is unavailable.

        Returns general route information (airlines, duration, price range) so
        the user gets a useful answer rather than an error message.
        """
        logger.info("FlightAgent._llm_fallback | query=%s", query[:80])
        try:
            messages = [
                SystemMessage(content=_GENERAL_FLIGHT_PROMPT),
                HumanMessage(content=query),
            ]
            return self._format_chain.invoke(messages).content
        except Exception as exc:
            logger.error("FlightAgent._llm_fallback | failed: %s", exc)
            return "Flight search is unavailable right now. Please check a booking site like Google Flights or Skyscanner for live prices."


def _extract_iata_and_date(query: str) -> dict:
    """Extract IATA codes and a YYYY-MM-DD date directly from the focused query string.

    _validate_flight_query writes queries like:
      "Find flights from MAA to BKK on 2026-06-29."
    This function reads them back out, providing a deterministic safety net
    that does not require another LLM call.
    """
    iata_codes = re.findall(r'\b([A-Z]{3})\b', query)
    date_match  = re.search(r'(\d{4}-\d{2}-\d{2})', query)
    # Filter common English 3-letter words that aren't IATA codes
    excluded = {"THE", "AND", "FOR", "FROM", "USE", "NOT", "OUT", "NOW"}
    iata_codes = [c for c in iata_codes if c not in excluded]

    return {
        "origin":      iata_codes[0] if len(iata_codes) >= 1 else "",
        "destination": iata_codes[1] if len(iata_codes) >= 2 else "",
        "date":        date_match.group(1) if date_match else "",
    }

