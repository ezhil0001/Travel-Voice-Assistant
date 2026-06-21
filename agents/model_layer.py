import logging
import os

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from config import settings

# Routing decisions and provider failures are logged here so on-call engineers
# can distinguish "Ollama was down" from "Groq rate-limited" without reading code.
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "test_results.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class ModelLayer:
    """Resilient LLM invocation with automatic retry and cloud fallback.

    The assistant runs Ollama locally to avoid per-token costs during normal
    operation. Ollama can be temporarily unreachable when the host machine is
    under load or the model is still loading — transient failures like these
    should never surface to the user. .with_retry() absorbs them silently.

    When Ollama is genuinely unavailable (host down, out of VRAM), Groq provides
    fast cloud inference as a last resort. .with_fallbacks() handles the switch
    automatically — the caller always gets a response or a clean error string.

    Both the primary model name and the Groq model are read from environment
    variables so the production instance can be reconfigured without a redeploy.
    """

    def __init__(self) -> None:
        # ChatOpenAI picks up OPENAI_BASE_URL and OPENAI_API_KEY from the environment.
        # settings.py points those at the local Ollama endpoint, so this file
        # has no knowledge of where Ollama is running.
        primary = ChatOpenAI(
            model=settings.OLLAMA_MODEL,
            temperature=0,
        )

        fallback = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
        )

        # Three attempts covers the typical Ollama cold-start window (~8 seconds)
        # without keeping the user waiting too long on a genuinely dead instance.
        primary_with_retry = primary.with_retry(
            stop_after_attempt=3,
        )

        # If all three Ollama attempts raise, LangChain passes the same request
        # to Groq. The caller sees a successful response — not a provider error.
        self._chain = primary_with_retry.with_fallbacks(
            fallbacks=[fallback],
            exceptions_to_handle=(Exception,),
        )

    def invoke(self, prompt: str) -> str:
        """Send a prompt and return the model's plain-text response.

        Retries and provider switching happen inside the chain — this method
        only needs to handle the case where everything fails simultaneously,
        which returns a safe error string rather than raising.

        Args:
            prompt: Fully assembled prompt string including system and user parts.

        Returns:
            Response text, or an error string if both providers are unavailable.
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
