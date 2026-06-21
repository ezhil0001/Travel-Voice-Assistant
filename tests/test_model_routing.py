# Verifies that ModelLayer correctly retries and falls back using LangChain's
# .with_retry() + .with_fallbacks() Runnable pattern.
# We patch the composed ._chain directly so tests are decoupled from which
# specific provider classes are used internally.

import logging
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)

from agents.model_layer import ModelLayer


def _make_layer_with_chain(chain_mock):
    """Build a ModelLayer and replace its internal chain with a mock."""
    with patch("agents.model_layer.ChatOpenAI"), \
         patch("agents.model_layer.ChatGroq"):
        layer = ModelLayer()
    layer._chain = chain_mock
    return layer


def test_chain_success_returns_content():
    """When the chain resolves, invoke() must return the .content string."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = MagicMock(content="Tokyo is beautiful")
    layer = _make_layer_with_chain(mock_chain)

    result = layer.invoke("Tell me about Tokyo")

    assert "Tokyo" in result
    mock_chain.invoke.assert_called_once_with("Tell me about Tokyo")
    log.info("PASS | test_chain_success_returns_content")


def test_chain_failure_returns_error_string():
    """When the chain itself raises (all retries + fallback exhausted), invoke()
    must return a string containing 'error' or 'unavailable' rather than raising.
    """
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("All providers down")
    layer = _make_layer_with_chain(mock_chain)

    result = layer.invoke("Anything")

    assert "error" in result.lower() or "unavailable" in result.lower()
    log.info("PASS | test_chain_failure_returns_error_string")


def test_with_retry_is_applied_to_primary():
    """ChatOpenAI must have .with_retry() called on it before .with_fallbacks()."""
    with patch("agents.model_layer.ChatOpenAI") as mock_openai_cls, \
         patch("agents.model_layer.ChatGroq"):

        mock_primary = MagicMock()
        mock_primary_with_retry = MagicMock()
        mock_chain = MagicMock()

        mock_openai_cls.return_value = mock_primary
        mock_primary.with_retry.return_value = mock_primary_with_retry
        mock_primary_with_retry.with_fallbacks.return_value = mock_chain

        layer = ModelLayer()

        # with_retry must have been called on the primary ChatOpenAI instance
        mock_primary.with_retry.assert_called_once()
        # with_fallbacks must have been called on the retry-wrapped primary
        mock_primary_with_retry.with_fallbacks.assert_called_once()
    log.info("PASS | test_with_retry_is_applied_to_primary")


def test_fallback_model_is_groq():
    """The fallback passed to .with_fallbacks() must be a ChatGroq instance."""
    with patch("agents.model_layer.ChatOpenAI") as mock_openai_cls, \
         patch("agents.model_layer.ChatGroq") as mock_groq_cls:

        mock_primary = MagicMock()
        mock_primary_with_retry = MagicMock()
        mock_groq_instance = MagicMock()

        mock_openai_cls.return_value = mock_primary
        mock_primary.with_retry.return_value = mock_primary_with_retry
        mock_groq_cls.return_value = mock_groq_instance

        ModelLayer()

        call_kwargs = mock_primary_with_retry.with_fallbacks.call_args
        fallbacks = call_kwargs.kwargs.get("fallbacks") or call_kwargs.args[0]
        assert mock_groq_instance in fallbacks, "Groq instance must be in fallbacks list"
    log.info("PASS | test_fallback_model_is_groq")


def test_ollama_model_comes_from_settings():
    """ChatOpenAI must be initialised with the model name from settings.OLLAMA_MODEL."""
    with patch("agents.model_layer.ChatOpenAI") as mock_openai_cls, \
         patch("agents.model_layer.ChatGroq"):
        mock_primary = MagicMock()
        mock_openai_cls.return_value = mock_primary
        mock_primary.with_retry.return_value = MagicMock()

        ModelLayer()

        call_kwargs = mock_openai_cls.call_args
        model_used = call_kwargs.kwargs.get("model") or call_kwargs.args[0]
        from config import settings
        assert model_used == settings.OLLAMA_MODEL, (
            f"Expected model '{settings.OLLAMA_MODEL}', got '{model_used}'"
        )
    log.info("PASS | test_ollama_model_comes_from_settings")
