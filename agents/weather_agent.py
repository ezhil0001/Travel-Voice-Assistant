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

_DISPLAY_PROMPT = (
    "You are a weather specialist. Format the tool result as a clean visual summary using markdown.\n\n"
    "Use this exact layout:\n"
    "**[City] — Current Conditions**\n\n"
    "| | |\n"
    "|---|---|\n"
    "| 🌡️ Temperature | X°C |\n"
    "| 🤔 Feels Like | X°C |\n"
    "| ☁️ Conditions | description |\n"
    "| 💧 Humidity | X% |\n\n"
    "**What to pack:** one practical sentence about clothing and essentials.\n\n"
    "Use only the data from the tool result. No extra text outside this layout."
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

                # Step 3 — Ask the model to format the raw data into a rich display response.
                # Send a clean 2-message request (no AIMessage with tool_calls) so
                # Groq doesn't reject the conversation with HTTP 400.
                format_messages = [
                    SystemMessage(content=_DISPLAY_PROMPT),
                    HumanMessage(content=f"Tool result: {tool_result}. Format this as a structured weather card."),
                ]
                try:
                    final = self._format_chain.invoke(format_messages)
                    if final.content and final.content.strip():
                        return final.content
                except Exception as fmt_exc:
                    logger.warning("WeatherAgent.run | format_chain failed: %s — using direct format", fmt_exc)

                # Direct fallback when format chain fails
                city   = tool_result.get("city", "your destination")
                temp   = tool_result.get("temperature", "N/A")
                desc   = tool_result.get("description", "N/A")
                feels  = tool_result.get("feels_like", "N/A")
                humid  = tool_result.get("humidity", "N/A")
                return (
                    f"**{city} — Current Conditions**\n\n"
                    f"| | |\n|---|---|\n"
                    f"| 🌡️ Temperature | {temp}°C |\n"
                    f"| 🤔 Feels Like | {feels}°C |\n"
                    f"| ☁️ Conditions | {desc} |\n"
                    f"| 💧 Humidity | {humid}% |\n\n"
                    f"**What to pack:** Light, breathable clothing recommended."
                )

            # LLM answered directly without a tool call.
            return response.content

        except Exception as exc:
            logger.error("WeatherAgent.run | failed: %s", exc)
            return "Sorry, I couldn't fetch the weather right now. Please try again."

