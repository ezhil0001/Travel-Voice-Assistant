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
                format_messages = messages + [
                    response,
                    HumanMessage(content=f"Tool result: {tool_result}. Now answer conversationally."),
                ]
                return self._format_chain.invoke(format_messages).content
            return response.content
        except Exception as exc:
            logger.error("AttractionsAgent.run | failed: %s", exc)
            return "Sorry, I couldn't find attractions right now. Please try again."

