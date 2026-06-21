from typing import Any, Callable

from middleware.base import AgentMiddleware

# Chains multiple AgentMiddleware instances into a single callable pipeline.
# The graph only needs to know about this one object — it doesn't have to
# wire each middleware hook individually.
#
# Execution order mirrors the LangChain hook pattern:
#   before_model    → runs in registration order (first-to-last)
#   wrap_model_call → nested, outermost middleware wraps all inner ones
#   after_model     → runs in reverse order (last-to-first)


class MiddlewarePipeline:
    """Composes a list of AgentMiddleware objects into an ordered execution chain.

    Usage:
        pipeline = MiddlewarePipeline([
            PreModelMiddleware(),
            DynamicPromptBuilder(),
            PostModelMiddleware(),
        ])

        state = pipeline.run_before_model(state)
        response = pipeline.run_wrap_model_call(request, model_handler)
        state = pipeline.run_after_model(state)
    """

    def __init__(self, middlewares: list[AgentMiddleware]) -> None:
        self._middlewares = middlewares

    def run_before_model(self, state: dict, **kwargs) -> dict:
        """Run all before_model hooks in registration order.

        Each hook receives the state produced by the previous one,
        so hooks form a sequential transformation chain.
        """
        for mw in self._middlewares:
            result = mw.before_model(state, **kwargs)
            if result is not None:
                state = result
        return state

    def run_wrap_model_call(
        self,
        request: dict,
        model_handler: Callable[[dict], Any],
        **kwargs,
    ) -> Any:
        """Nest all wrap_model_call hooks around the model invocation.

        The first registered middleware is the outermost wrapper — it calls
        the second, which calls the third, ... which finally calls the model.
        """
        handler: Callable[[dict], Any] = model_handler
        for mw in reversed(self._middlewares):
            def make_handler(m: AgentMiddleware, h: Callable) -> Callable:
                return lambda req, **kw: m.wrap_model_call(req, h, **kw)
            handler = make_handler(mw, handler)

        return handler(request, **kwargs)

    def run_after_model(self, state: dict, **kwargs) -> dict:
        """Run all after_model hooks in reverse registration order.

        Reversing matches the convention where the innermost middleware
        (closest to the model) post-processes first.
        """
        for mw in reversed(self._middlewares):
            result = mw.after_model(state, **kwargs)
            if result is not None:
                state = result
        return state
