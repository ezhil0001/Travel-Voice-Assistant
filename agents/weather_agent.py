# Weather sub-agent — extends BaseAgent which handles chain construction,
# tool binding, and retry/fallback. This class owns only its tool, its
# system prompt, and its result-formatting logic. No middleware here.

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
from tools.weather_tool import get_weather

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a weather specialist for a voice travel assistant. "
    "Extract the city from the user's question, call get_weather, "
    "and respond with temperature, conditions, and 1-2 packing tips. "
    "Keep it under 2 sentences. No markdown."
)


class WeatherAgent(BaseAgent):
    """Resolves weather queries using LangChain tool binding + Runnable chain.

    Extends BaseAgent which builds the bound_chain and format_chain.
    run() is the only method here — strictly:
        tool invocation → LLM formatting → return plain string.
    Middleware orchestration is handled by ServiceLayer, not this class.
    """

    def __init__(self) -> None:
        # BaseAgent builds _bound_chain, _format_chain, and stores _tool.
        super().__init__(tool=get_weather)

    def run(self, query: str) -> str:
        """Tool invocation → LLM formatting → return plain string.

        ServiceLayer handles middleware before and after this call.
        This method is strictly: bind_tools response → execute tool → format.
        """
        logger.info("WeatherAgent.run | query=%s", query)
        try:
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]

            # Step 1 — Let the bound LLM decide to call the tool.
            response = self._bound_chain.invoke(messages)

            # Step 2 — If the LLM issued a tool call, execute it and format.
            if response.tool_calls:
                tool_call = response.tool_calls[0]
                tool_result = get_weather.invoke(tool_call["args"])

                if "error" in tool_result:
                    return f"Sorry, I couldn't fetch the weather. {tool_result['error']}"

                # Step 3 — Ask the model to format the raw data into a voice response.
                format_messages = messages + [
                    response,
                    HumanMessage(content=f"Tool result: {tool_result}. Now answer conversationally."),
                ]
                final = self._format_chain.invoke(format_messages)
                return final.content

            # LLM answered directly without a tool call.
            return response.content

        except Exception as exc:
            logger.error("WeatherAgent.run | failed: %s", exc)
            return "Sorry, I couldn't fetch the weather right now. Please try again."

