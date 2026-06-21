# ServiceLayer is the single coordination point between the middleware pipeline
# and the agent that handles a user's request.
#
# Keeping this boundary explicit means we can swap middleware behaviour (e.g.
# change how input is cleaned, or how long responses can be) without touching
# any agent file, and we can add new agents without touching any middleware file.
#
# Execution order:
#   1. before_model  — PreModelMiddleware cleans the raw STT transcript,
#                      detects the destination city, and injects recent history
#   2. wrap_model_call — DynamicPromptBuilder assembles the system prompt just
#                        before the agent calls the LLM, so the instruction set
#                        reflects the current agent domain (weather vs. flights etc.)
#   3. after_model   — PostModelMiddleware strips markdown and truncates the
#                      response so it reads naturally when converted to speech

import logging
from agents.base_agent import BaseAgent
from middleware.pre_model import PreModelMiddleware
from middleware.dynamic_prompt import DynamicPromptBuilder
from middleware.post_model import PostModelMiddleware
from middleware.pipeline import MiddlewarePipeline

logger = logging.getLogger(__name__)


class ServiceLayer:
    """Runs the middleware pipeline around an agent's core logic.

    The pipeline is composed once at startup and reused across every request.
    Middleware instances are stateless per call, so sharing them is safe.

    Agents receive only the cleaned user input — they have no visibility into
    conversation history, raw STT text, or session metadata. This prevents
    an agent from accidentally depending on state it shouldn't own.
    """

    def __init__(self) -> None:
        self._pipeline = MiddlewarePipeline([
            PreModelMiddleware(),    # sanitises STT transcript, detects city
            DynamicPromptBuilder(), # assembles domain-specific system prompt
            PostModelMiddleware(),   # strips markdown, caps length for TTS
        ])

    def execute(
        self,
        agent: BaseAgent,
        agent_type: str,
        query: str,
        history: list | None = None,
    ) -> str:
        """Run a user query through the full middleware → agent → middleware cycle.

        Args:
            agent:      The domain agent that will handle this request.
            agent_type: One of weather/flight/attractions/currency/timezone/general.
                        Controls which instruction block the prompt builder selects.
            query:      Raw user input — may still contain STT artefacts.
            history:    Recent conversation turns for pronoun resolution context.

        Returns:
            A clean, TTS-ready string with no markdown or URLs.
        """
        if history is None:
            history = []

        # Stage 1 — clean and enrich the incoming state.
        state: dict = {
            "user_input": query,
            "conversation_history": history,
        }
        state = self._pipeline.run_before_model(state)

        cleaned_input: str = state["cleaned_input"]
        context: dict = {"history": state.get("context", [])}

        logger.info(
            "ServiceLayer.execute | agent=%s | city=%s | cleaned=%s",
            agent_type,
            state.get("metadata", {}).get("detected_city"),
            cleaned_input,
        )

        # Stage 2 — let DynamicPromptBuilder inject the system prompt, then
        # hand off only the cleaned text to the agent. The agent has no access
        # to the surrounding state dict — only the string it needs to act on.
        def model_handler(request: dict) -> str:
            return agent.run(request["cleaned_input"])

        request: dict = {
            "cleaned_input": cleaned_input,
            "agent_type": agent_type,
            "context": context,
        }
        raw_response: str = self._pipeline.run_wrap_model_call(request, model_handler)

        # Stage 3 — strip any markdown the LLM emitted and cap the length so
        # the TTS engine doesn't produce a multi-minute audio response.
        state["raw_response"] = raw_response
        state = self._pipeline.run_after_model(state)

        clean_response: str = state.get("cleaned_response", raw_response)
        logger.info("ServiceLayer.execute | response_length=%d", len(clean_response))
        return clean_response
