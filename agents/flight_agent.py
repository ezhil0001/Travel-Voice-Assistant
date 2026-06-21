# Flight sub-agent — extends BaseAgent which handles chain construction,
# tool binding, and retry/fallback. This class owns only its tool, its
# system prompt, and its result-formatting logic. No middleware here.

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
from tools.flight_tool import get_flights

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a flight search specialist for a voice travel assistant. "
    "Extract the departure city, destination city, and travel date from the query. "
    "Call get_flights using 3-letter IATA codes (e.g. New York → JFK, Tokyo → NRT). "
    "Return the cheapest option with price, stops, and airline. One sentence. No markdown."
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

        ServiceLayer handles middleware before and after this call.
        origin/destination/date hints are injected by the supervisor when
        it has already resolved IATA codes from the conversation context.
        """
        logger.info("FlightAgent.run | %s → %s on %s", origin, destination, date)
        try:
            context = ""
            if origin and destination and date:
                context = f" (Use origin={origin}, destination={destination}, date={date})"

            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=query + context),
            ]

            response = self._bound_chain.invoke(messages)

            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_result = self._tool.invoke(tool_call["args"])

                if isinstance(tool_result, list) and tool_result and "error" in tool_result[0]:
                    return f"Sorry, I couldn't find flights. {tool_result[0]['error']}"

                format_messages = messages + [
                    response,
                    HumanMessage(content=f"Tool result: {tool_result}. Now answer conversationally."),
                ]
                final = self._format_chain.invoke(format_messages)
                return final.content

            return response.content

        except Exception as exc:
            logger.error("FlightAgent.run | failed: %s", exc)
            return "Sorry, I couldn't search for flights right now. Please try again."

