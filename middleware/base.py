from abc import ABC
from typing import Any, Callable, Optional


# Abstract base that mirrors LangChain's AgentMiddleware hook pattern.
# Each concrete middleware only overrides the hooks it needs — the rest
# are no-ops by default so the pipeline never breaks when a hook is missing.


class AgentMiddleware(ABC):
    """Lifecycle hooks that intercept the agent execution flow.

    Subclasses override one or more of three hook methods:

        before_model    — fires before the LLM receives any input.
                          Use it for input cleaning, validation, context injection.

        wrap_model_call — wraps the actual model invocation.
                          Use it to modify the prompt or implement retry logic.

        after_model     — fires after the LLM returns a response.
                          Use it for output formatting, safety filtering, logging.

    Hooks that return None are treated as no-ops by the pipeline.
    """

    def before_model(self, state: dict, **kwargs) -> Optional[dict]:
        """Called before every LLM invocation.

        Args:
            state: Mutable dict representing the current agent state.
                   At minimum contains 'user_input' and 'conversation_history'.

        Returns:
            An updated state dict, or None to leave state unchanged.
        """
        return None

    def wrap_model_call(
        self,
        request: dict,
        handler: Callable[[dict], Any],
        **kwargs,
    ) -> Any:
        """Wraps the model call — lets middleware modify the prompt or retry.

        Args:
            request: Dict with at least 'prompt' and 'messages' keys.
            handler: The next callable in the chain (model or next middleware).

        Returns:
            The model response (modified or original).
        """
        return handler(request)

    def after_model(self, state: dict, **kwargs) -> Optional[dict]:
        """Called after the LLM returns a response.

        Args:
            state: Agent state now also containing 'raw_response'.

        Returns:
            An updated state dict, or None to leave state unchanged.
        """
        return None
