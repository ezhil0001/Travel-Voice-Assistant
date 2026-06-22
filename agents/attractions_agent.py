# Attractions sub-agent — extends BaseAgent which handles chain construction,
# tool binding, and retry/fallback. This class owns only its tool, its
# system prompt, and its result-formatting logic. No middleware here.

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
from tools.attractions_tool import get_attractions

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a tourism expert for a voice travel assistant. "
    "Extract the city from the user's question, call get_attractions, "
    "and mention the top 3 places with a brief one-word description each. "
    "Format: 'In [city] you should visit [place1] ([word]), [place2] ([word]), "
    "and [place3] ([word]).' No markdown. One sentence only."
)


class AttractionsAgent(BaseAgent):
    """Returns top tourist attractions using LangChain tool binding + Runnable chain.

    Extends BaseAgent which builds the bound_chain and format_chain.
    run() strictly handles: tool invocation → LLM formatting → return string.
    """

    def __init__(self) -> None:
        super().__init__(tool=get_attractions)

    def run(self, query: str) -> str:
        """Tool invocation → LLM formatting → return plain string."""
        logger.info("AttractionsAgent.run | query=%s", query)
        try:
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
            response = self._bound_chain.invoke(messages)
            if response.tool_calls:
                tool_result = self._tool.invoke(response.tool_calls[0]["args"])
                if isinstance(tool_result, list) and tool_result and "error" in tool_result[0]:
                    return "Sorry, I couldn't find attractions right now."
                format_messages = [
                    SystemMessage(content=_SYSTEM_PROMPT),
                    HumanMessage(content=f"Tool result: {tool_result}. Give a voice-friendly attractions summary."),
                ]
                try:
                    formatted = self._format_chain.invoke(format_messages).content
                except Exception as fmt_exc:
                    logger.warning("AttractionsAgent.run | format_chain failed: %s — using direct fallback", fmt_exc)
                    formatted = ""
                # If the format chain returns empty (model refused or over-thought),
                # build a plain-text fallback directly from the tool result.
                if not formatted or not formatted.strip():
                    logger.warning("AttractionsAgent.run | format_chain returned empty — using direct fallback")
                    names = [p.get("name", "Unknown") for p in tool_result if isinstance(p, dict) and "name" in p]
                    if names:
                        city_arg = response.tool_calls[0]["args"].get("city", "the destination")
                        return f"In {city_arg} you should visit {', '.join(names[:3])}."
                    return "I found some attractions but couldn't format them. Please ask again."
                return formatted
            # No tool call — the LLM answered from its own knowledge (acceptable for well-known cities)
            if response.content and response.content.strip():
                return response.content
            return "I couldn't identify a destination from your query. Please mention the city name."
        except Exception as exc:
            logger.error("AttractionsAgent.run | failed: %s", exc)
            return "Sorry, I couldn't find attractions right now. Please try again."

