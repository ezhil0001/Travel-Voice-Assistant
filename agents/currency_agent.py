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
    "Read the user's query carefully and extract:\n"
    "  1. from_c: the SOURCE currency code (e.g. INR if the user mentions India or Chennai, "
    "USD if they mention the USA). Infer from the departure location if not stated explicitly.\n"
    "  2. to_c: the TARGET/destination currency code (e.g. THB for Thailand, JPY for Japan).\n"
    "  3. amount: the total amount to convert. If the user mentions a number of days and a "
    "daily budget context, estimate a realistic travel budget (e.g. 5 days in Thailand → "
    "25000 INR total). If no amount is mentioned, use 1000 as a sensible default.\n"
    "Then call get_currency with those three values. "
    "State the converted amount and add one practical sentence about what it buys locally. "
    "No markdown. Two sentences max."
)

_DISPLAY_PROMPT = (
    "You are a currency specialist. Format the tool result as a clean visual summary using markdown.\n\n"
    "Use this exact layout:\n"
    "**Currency Conversion — [from_currency] → [to_currency]**\n\n"
    "| | |\n"
    "|---|---|\n"
    "| 💵 Amount | X [from_c] |\n"
    "| 💱 Converted | X [to_c] |\n"
    "| 📈 Exchange Rate | 1 [from_c] = X [to_c] |\n\n"
    "**Purchasing power:** one practical sentence about what this amount buys locally "
    "(e.g. meals, transport, accommodation).\n\n"
    "Use only the data from the tool result. Do not fabricate rates."
)


class CurrencyAgent(BaseAgent):
    """Converts currencies using LangChain tool binding + Runnable chain.

    Extends BaseAgent which builds the bound_chain and format_chain.
    run() strictly handles: tool invocation → LLM formatting → return string.
    """

    def __init__(self) -> None:
        super().__init__(tool=get_currency)

    def run(self, query: str) -> str:
        """Tool invocation → LLM formatting → return plain string.

        The LLM extracts from_currency, to_currency, and amount entirely from
        the query text. No default currency values are injected here — doing so
        caused the agent to call get_currency with USD→USD regardless of what
        the user actually asked.
        """
        logger.info("CurrencyAgent.run | query=%s", query[:80])
        try:
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
            response = self._bound_chain.invoke(messages)
            if response.tool_calls:
                args = response.tool_calls[0]["args"]
                logger.info("CurrencyAgent.run | tool_args=%s", args)
                tool_result = self._tool.invoke(args)
                if "error" in tool_result:
                    return f"Sorry, I couldn't fetch the exchange rate. {tool_result['error']}"
                format_messages = [
                    SystemMessage(content=_DISPLAY_PROMPT),
                    HumanMessage(content=f"Tool result: {tool_result}. Format this as a structured currency card."),
                ]
                try:
                    formatted = self._format_chain.invoke(format_messages).content
                    if formatted and formatted.strip():
                        return formatted
                except Exception as fmt_exc:
                    logger.warning("CurrencyAgent.run | format_chain failed: %s — using direct format", fmt_exc)

                # Direct fallback when format chain fails
                rate    = tool_result.get("conversion_rate", "")
                result  = tool_result.get("conversion_result", "")
                from_c  = args.get("from_c", "")
                to_c    = args.get("to_c", "")
                amount  = args.get("amount", "")
                return (
                    f"**Currency Conversion — {from_c} → {to_c}**\n\n"
                    f"| | |\n|---|---|\n"
                    f"| 💵 Amount | {amount} {from_c} |\n"
                    f"| 💱 Converted | {result} {to_c} |\n"
                    f"| 📈 Exchange Rate | 1 {from_c} = {rate} {to_c} |"
                )
            return response.content
        except Exception as exc:
            logger.error("CurrencyAgent.run | failed: %s", exc)
            return "Sorry, I couldn't fetch the exchange rate right now. Please try again."

