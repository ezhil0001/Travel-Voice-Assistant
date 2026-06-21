import logging
import os

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from config import settings

# All model routing events are written here so failures are easy to trace
# without attaching a debugger.
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "test_results.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class ModelLayer:
    """LLM invocation layer using the official LangChain Runnable resilience pattern.

    Resilience is built from two composable Runnable wrappers:

        .with_retry()      — mirrors ModelRetryMiddleware from the blueprint.
                             Retries transient failures (timeouts, 500s, rate limits)
                             up to 3 times with exponential backoff before giving up.

        .with_fallbacks()  — mirrors ModelFallbackMiddleware from the blueprint.
                             If all retries on the primary model are exhausted,
                             the chain automatically tries the next model in the list.

    Primary  : Ollama / qwen3.5:122b — accessed via its OpenAI-compatible REST API.
               The endpoint and API key are read from OPENAI_BASE_URL / OPENAI_API_KEY
               environment variables (set by config/settings.py), so model_layer.py
               contains zero hardcoded URLs.

    Fallback : Groq / openai/gpt-oss-120b — fast cloud inference when Ollama is
               unreachable, overloaded, or the local GPU is busy.

    The composed chain is built once in __init__ and reused on every invoke() call.
    """

    def __init__(self) -> None:
        # ChatOpenAI reads OPENAI_BASE_URL and OPENAI_API_KEY automatically from
        # the environment — settings.py sets those to the Ollama endpoint so no
        # URL string needs to appear anywhere in this file.
        primary = ChatOpenAI(
            model=settings.OLLAMA_MODEL,
            temperature=0,
        )

        fallback = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
        )

        # .with_retry() wraps the primary with exponential backoff retry logic.
        # stop_after_attempt=3 means: attempt 1 immediately, attempt 2 after 2 s,
        # attempt 3 after 4 s — matching the blueprint's documented behaviour.
        primary_with_retry = primary.with_retry(
            stop_after_attempt=3,
        )

        # .with_fallbacks() chains the retry-wrapped primary with the fallback model.
        # If primary_with_retry raises after exhausting all attempts, LangChain
        # automatically invokes `fallback` — the agent never sees a provider error.
        self._chain = primary_with_retry.with_fallbacks(
            fallbacks=[fallback],
            exceptions_to_handle=(Exception,),
        )

    def invoke(self, prompt: str) -> str:
        """Send a prompt through the resilient model chain and return plain text.

        The chain handles retries and failover internally. This method only needs
        to call the chain and extract the response content.

        Args:
            prompt: The fully assembled prompt string to send to the LLM.

        Returns:
            The model's response as a plain string, or an error message if
            both primary and fallback providers fail.
        """
        try:
            logger.info("ModelLayer.invoke | primary=%s | fallback=%s",
                        settings.OLLAMA_MODEL, settings.GROQ_MODEL)
            response = self._chain.invoke(prompt)
            logger.info("ModelLayer.invoke | succeeded")
            return response.content
        except Exception as exc:
            logger.error("ModelLayer.invoke | all providers failed: %s", exc)
            return "error: All models unavailable. Please try again later."
