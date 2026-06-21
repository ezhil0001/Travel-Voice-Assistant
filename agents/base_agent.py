# All five domain agents (weather, flight, attractions, currency, timezone)
# share the same LLM wiring — primary Ollama model bound to a specific tool,
# with retry and Groq fallback. That wiring lives here so each agent file
# stays focused on its own tool, system prompt, and response format.
#
# The ABC enforces run() on every subclass so the ServiceLayer can call
# any agent without knowing its type — it just calls agent.run(query).

import logging
from abc import ABC, abstractmethod
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from config import settings

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Shared chain construction for all travel domain agents.

    Builds two Runnable chains at init time and exposes them as protected
    attributes so subclasses can use them without duplicating the retry/fallback
    setup across five files:

        _bound_chain   — LLM with the domain tool registered. Used for the
                         first model call where the LLM decides whether to invoke
                         the tool and with what arguments.

        _format_chain  — Plain LLM without tools. Used for the second call
                         where the raw tool result is turned into a spoken sentence.

        _tool          — The registered LangChain tool, stored so subclasses can
                         call _tool.invoke(args) directly after extracting args
                         from the bound chain's tool_calls response.

    Subclasses must implement run() and must not touch middleware state.
    """

    def __init__(self, tool: BaseTool) -> None:
        primary = ChatOpenAI(model=settings.OLLAMA_MODEL, temperature=0)
        fallback = ChatGroq(model=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY)

        # The LLM needs the tool schema registered so it knows when and how to
        # call it. The fallback also gets the same tool bound so behaviour is
        # consistent regardless of which provider handles the request.
        self._bound_chain = (
            primary.bind_tools([tool])
            .with_retry(stop_after_attempt=3)
            .with_fallbacks(
                fallbacks=[fallback.bind_tools([tool])],
                exceptions_to_handle=(Exception,),
            )
        )

        # A separate chain without tools for the formatting step — we don't want
        # the LLM to trigger another tool call when we're just asking it to turn
        # a JSON blob into a readable sentence.
        self._format_chain = (
            primary
            .with_retry(stop_after_attempt=3)
            .with_fallbacks(
                fallbacks=[fallback],
                exceptions_to_handle=(Exception,),
            )
        )

        self._tool = tool

    @abstractmethod
    def run(self, query: str) -> str:
        """Execute the domain logic and return a plain, spoken-word response.

        Contracts for all implementations:
            - Receives a clean string (STT artefacts already removed upstream).
            - Returns plain text with no markdown — the response goes to TTS.
            - Catches all exceptions internally and returns a human-readable
              fallback rather than raising.
        """
