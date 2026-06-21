# Timezone sub-agent — extends BaseAgent which handles chain construction,
# tool binding, and retry/fallback. This class owns only its tool, its
# system prompt, and its result-formatting logic. No middleware here.

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
from tools.timezone_tool import get_timezone

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a timezone expert for a voice travel assistant. "
    "Extract the city or timezone from the query, call get_timezone using "
    "the correct IANA timezone string (e.g. 'Asia/Tokyo', 'Europe/Paris'). "
    "Report the current local time and state how many hours ahead or behind "
    "US Eastern Time (UTC-5) it is. One sentence. No markdown."
)


class TimezoneAgent(BaseAgent):
    """Reports local time at a destination using LangChain tool binding + Runnable chain.

    Extends BaseAgent which builds the bound_chain and format_chain.
    run() strictly handles: tool invocation → LLM formatting → return string.
    """

    def __init__(self) -> None:
        super().__init__(tool=get_timezone)

    def run(self, query: str, timezone: str = "UTC") -> str:
        """Tool invocation → LLM formatting → return plain string."""
        logger.info("TimezoneAgent.run | timezone=%s", timezone)
        try:
            context = f" (Use timezone={timezone})" if timezone and timezone != "UTC" else ""
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=query + context),
            ]
            response = self._bound_chain.invoke(messages)
            if response.tool_calls:
                tool_result = self._tool.invoke(response.tool_calls[0]["args"])
                if "error" in tool_result:
                    return f"Sorry, I couldn't fetch the timezone information. {tool_result['error']}"
                format_messages = messages + [
                    response,
                    HumanMessage(content=f"Tool result: {tool_result}. Now answer conversationally."),
                ]
                return self._format_chain.invoke(format_messages).content
            return response.content
        except Exception as exc:
            logger.error("TimezoneAgent.run | failed: %s", exc)
            return "Sorry, I couldn't fetch the timezone information right now. Please try again."

