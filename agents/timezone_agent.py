# Timezone sub-agent — extends BaseAgent which handles chain construction,
# tool binding, and retry/fallback. This class owns only its tool, its
# system prompt, and its result-formatting logic. No middleware here.

import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent
from tools.timezone_tool import get_timezone

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a timezone expert for a voice travel assistant. "
    "The query may contain an explicit IANA timezone string after 'Use timezone'. "
    "If present, use that string EXACTLY as the argument to get_timezone — do NOT guess. "
    "If no explicit timezone is given, derive the correct IANA string from the city name. "
    "Report the current local time and how many hours ahead or behind US Eastern Time it is. "
    "One sentence. No markdown."
)

_DISPLAY_PROMPT = (
    "You are a timezone specialist. Format the tool result as a clean visual summary using markdown.\n\n"
    "Use this exact layout:\n"
    "**Local Time — [City / Timezone]**\n\n"
    "| | |\n"
    "|---|---|\n"
    "| 🕐 Local Time | HH:MM (AM/PM) |\n"
    "| 📅 Date | Day, DD Month YYYY |\n"
    "| 🌍 Timezone | IANA zone name |\n"
    "| ⏱️ vs. UTC | +/- N hours |\n\n"
    "**Tip:** one short practical sentence about the time difference "
    "(e.g. what it means for calling home or planning activities).\n\n"
    "Use only the data from the tool result. Format the time as 12-hour with AM/PM."
)


class TimezoneAgent(BaseAgent):
    """Reports local time at a destination using LangChain tool binding + Runnable chain.

    Extends BaseAgent which builds the bound_chain and format_chain.
    run() strictly handles: tool invocation → LLM formatting → return string.
    """

    def __init__(self) -> None:
        super().__init__(tool=get_timezone)

    def run(self, query: str, timezone: str = "UTC") -> str:
        """Tool invocation → LLM formatting → return plain string.

        If _validate_timezone_query embedded an explicit IANA timezone in the
        focused query (e.g. "Use timezone Asia/Bangkok"), extract it here and
        use it directly as the tool argument rather than relying on the LLM to
        re-derive it — eliminates the UTC fallback bug.
        """
        # Extract explicit IANA timezone injected by _validate_timezone_query
        iana_hint = _extract_iana(query) or (timezone if timezone != "UTC" else "")
        logger.info("TimezoneAgent.run | timezone=%s | iana_hint=%s", timezone, iana_hint)

        try:
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
            response = self._bound_chain.invoke(messages)
            if response.tool_calls:
                args = response.tool_calls[0]["args"]
                # Override LLM-chosen timezone with our validated hint when available
                if iana_hint:
                    args = {**args, "timezone": iana_hint}
                logger.info("TimezoneAgent.run | tool_args=%s", args)
                tool_result = self._tool.invoke(args)
                if "error" in tool_result:
                    return f"Sorry, I couldn't fetch the timezone information. {tool_result['error']}"
                format_messages = [
                    SystemMessage(content=_DISPLAY_PROMPT),
                    HumanMessage(content=f"Tool result: {tool_result}. Format this as a structured timezone card."),
                ]
                return self._format_chain.invoke(format_messages).content
            return response.content
        except Exception as exc:
            logger.error("TimezoneAgent.run | failed: %s", exc)
            # If we already have the IANA string (injected by the focused-query
            # builder), call the tool directly and format the result ourselves
            # rather than returning a sorry message — the LLM failure doesn't
            # mean the timezone data is unavailable.
            if iana_hint:
                try:
                    tool_result = self._tool.invoke({"timezone": iana_hint})
                    if "error" not in tool_result:
                        dt       = tool_result.get("datetime", "")
                        time_str = dt[11:16] if len(dt) >= 16 else ""
                        tz       = tool_result.get("timezone", iana_hint)
                        utc_off  = tool_result.get("utc_offset", "")
                        city     = tz.split("/")[-1].replace("_", " ")
                        if time_str:
                            return (
                                f"**Local Time — {city}**\n\n"
                                f"| | |\n|---|---|\n"
                                f"| 🕐 Local Time | {time_str} |\n"
                                f"| 🌍 Timezone | {tz} |\n"
                                f"| ⏱️ vs. UTC | {utc_off} |"
                            )
                except Exception as tool_exc:
                    logger.error("TimezoneAgent.run | direct tool call also failed: %s", tool_exc)
            return "Sorry, I couldn't fetch the timezone information right now. Please try again."


def _extract_iana(query: str) -> str:
    """Pull out the IANA timezone string injected by _validate_timezone_query.

    Matches patterns like: 'Use timezone Asia/Bangkok' or 'timezone: Asia/Bangkok'.
    Returns empty string if no match.
    """
    match = re.search(r'[Uu]se\s+timezone\s+([\w/]+)', query)
    if match:
        return match.group(1)
    match = re.search(r'timezone[:\s]+([A-Za-z]+/[A-Za-z_]+)', query)
    if match:
        return match.group(1)
    return ""

