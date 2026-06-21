# Currency sub-agent — extends BaseAgent which handles chain construction,
# tool binding, and retry/fallback. This class owns only its tool, its
# system prompt, and its result-formatting logic. No middleware here.

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
from tools.currency_tool import get_currency

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a currency conversion specialist for a voice travel assistant. "
    "Extract from_currency, to_currency, and amount from the query. "
    "Call get_currency and state the converted amount. "
    "Add one practical context sentence like 'that covers a nice dinner'. "
    "No markdown. Two sentences max."
)


class CurrencyAgent(BaseAgent):
    """Converts currencies using LangChain tool binding + Runnable chain.

    Extends BaseAgent which builds the bound_chain and format_chain.
    run() strictly handles: tool invocation → LLM formatting → return string.
    """

    def __init__(self) -> None:
        super().__init__(tool=get_currency)

    def run(self, query: str, from_currency: str = "USD",
            to_currency: str = "USD", amount: float = 1.0) -> str:
        """Tool invocation → LLM formatting → return plain string."""
        logger.info("CurrencyAgent.run | %s %s → %s", amount, from_currency, to_currency)
        try:
            context = ""
            if from_currency and to_currency and amount:
                context = f" (from_c={from_currency}, to_c={to_currency}, amount={amount})"
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=query + context),
            ]
            response = self._bound_chain.invoke(messages)
            if response.tool_calls:
                tool_result = self._tool.invoke(response.tool_calls[0]["args"])
                if "error" in tool_result:
                    return f"Sorry, I couldn't fetch the exchange rate. {tool_result['error']}"
                format_messages = messages + [
                    response,
                    HumanMessage(content=f"Tool result: {tool_result}. Now answer conversationally."),
                ]
                return self._format_chain.invoke(format_messages).content
            return response.content
        except Exception as exc:
            logger.error("CurrencyAgent.run | failed: %s", exc)
            return "Sorry, I couldn't fetch the exchange rate right now. Please try again."

