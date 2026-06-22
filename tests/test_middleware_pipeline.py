# Tests the middleware stack — both the direct .process() helpers and the
# LangChain-style hook API (before_model / wrap_model_call / after_model).
# Keeping hook tests separate from helper tests makes failures easier to pinpoint.
import logging

log = logging.getLogger(__name__)

from middleware.pre_model import PreModelMiddleware
from middleware.dynamic_prompt import DynamicPromptBuilder
from middleware.post_model import PostModelMiddleware
from middleware.pipeline import MiddlewarePipeline


# ── Pre-model tests ────────────────────────────────────────────────────────────

def test_pre_model_cleans_whitespace():
    """Extra spaces from STT pauses must be collapsed to single spaces."""
    mw = PreModelMiddleware()
    result = mw.process("  what is the weather in   Tokyo  ", [])
    assert result["cleaned_input"] == "what is the weather in Tokyo"
    log.info("PASS | test_pre_model_cleans_whitespace")


def test_pre_model_detects_city():
    """City detection should return the properly-cased city name."""
    mw = PreModelMiddleware()
    result = mw.process("what is the weather in Tokyo", [])
    assert result["metadata"]["detected_city"] == "Tokyo"
    log.info("PASS | test_pre_model_detects_city")


def test_pre_model_no_city():
    """When no known city is mentioned, detected_city must be None."""
    mw = PreModelMiddleware()
    result = mw.process("what can you help me with?", [])
    assert result["metadata"]["detected_city"] is None
    log.info("PASS | test_pre_model_no_city")


def test_pre_model_context_slicing():
    """Only the last 3 turns of history should be injected into context."""
    mw = PreModelMiddleware()
    history = [{"role": "user", "content": f"msg {i}"} for i in range(6)]
    result = mw.process("hello", history)
    assert len(result["context"]) == 3
    assert result["context"][0]["content"] == "msg 3"
    log.info("PASS | test_pre_model_context_slicing")


def test_pre_model_metadata_keys():
    """Metadata dict must always contain timestamp, input_length, detected_city."""
    mw = PreModelMiddleware()
    result = mw.process("fly to Paris", [])
    for key in ("timestamp", "input_length", "detected_city"):
        assert key in result["metadata"], f"Missing metadata key: {key}"
    log.info("PASS | test_pre_model_metadata_keys")


# ── Dynamic prompt tests ───────────────────────────────────────────────────────

def test_dynamic_prompt_contains_base():
    """Base system prompt must always be present regardless of agent type."""
    builder = DynamicPromptBuilder()
    prompt = builder.build("anything", {}, "general")
    assert "travel planning assistant" in prompt.lower()
    log.info("PASS | test_dynamic_prompt_contains_base")


def test_dynamic_prompt_weather_instruction():
    """Weather agent prompt must mention packing guidance."""
    builder = DynamicPromptBuilder()
    prompt = builder.build("weather in Paris", {}, "weather")
    assert "pack" in prompt.lower()
    log.info("PASS | test_dynamic_prompt_weather_instruction")


def test_dynamic_prompt_flight_iata_constraint():
    """Flight prompt must include the IATA code constraint for the tool."""
    builder = DynamicPromptBuilder()
    prompt = builder.build("flights from New York to Tokyo", {}, "flight")
    assert "IATA" in prompt or "iata" in prompt.lower() or "JFK" in prompt
    log.info("PASS | test_dynamic_prompt_flight_iata_constraint")


def test_dynamic_prompt_unknown_type_fallback():
    """An unrecognised agent type should fall back to the general instruction block."""
    builder = DynamicPromptBuilder()
    prompt = builder.build("random query", {}, "unknown_agent")
    assert "conversationally" in prompt.lower()
    log.info("PASS | test_dynamic_prompt_unknown_type_fallback")


def test_dynamic_prompt_injects_history():
    """Recent conversation turns should appear inside the assembled prompt."""
    builder = DynamicPromptBuilder()
    context = {"history": [{"role": "user", "content": "I want to visit Rome"}]}
    prompt = builder.build("what to pack?", context, "weather")
    assert "Rome" in prompt
    log.info("PASS | test_dynamic_prompt_injects_history")


# ── Post-model tests ───────────────────────────────────────────────────────────

def test_post_model_removes_bold_markdown():
    """Bold markdown tokens must be stripped from LLM output."""
    mw = PostModelMiddleware()
    result = mw.process("**Tokyo** is a great city to visit.")
    assert "**" not in result
    assert "Tokyo" in result
    log.info("PASS | test_post_model_removes_bold_markdown")


def test_post_model_removes_headers():
    """Markdown headers (##) must not appear in TTS output."""
    mw = PostModelMiddleware()
    result = mw.process("## Top Attractions\nShibuya and Shinjuku.")
    assert "##" not in result
    log.info("PASS | test_post_model_removes_headers")


def test_post_model_removes_urls():
    """Bare URLs should be stripped entirely — they sound terrible when spoken."""
    mw = PostModelMiddleware()
    raw = "**Tokyo** is great! Visit ##Shibuya and [click here](http://x.com)"
    result = mw.process(raw)
    assert "**" not in result
    assert "http" not in result
    log.info("PASS | test_post_model_removes_urls")


def test_post_model_truncates_long_response():
    """Responses over POST_MODEL_MAX_CHARS must be trimmed and end with the follow-up cue."""
    mw = PostModelMiddleware()
    # Build a string guaranteed to exceed whatever the current max is
    long_text = "a" * (mw._max_chars + 500)
    result = mw.process(long_text)
    # Trimmed text + space + suffix must stay within max_chars + len(suffix) + 1
    assert len(result) <= mw._max_chars + len(mw._overflow_suffix) + 1
    assert "just ask" in result
    log.info("PASS | test_post_model_truncates_long_response")


def test_post_model_short_response_unchanged():
    """Short clean responses should pass through without modification."""
    mw = PostModelMiddleware()
    short = "Tokyo is beautiful in spring."
    result = mw.process(short)
    assert result == short
    log.info("PASS | test_post_model_short_response_unchanged")


# ── Hook-pattern tests (LangChain-style middleware API) ────────────────────────

def test_pre_model_before_model_hook():
    """before_model hook must write cleaned_input and metadata into state."""
    mw = PreModelMiddleware()
    state = {"user_input": "  weather in  Tokyo  ", "conversation_history": []}
    result = mw.before_model(state)
    assert result["cleaned_input"] == "weather in Tokyo"
    assert result["metadata"]["detected_city"] == "Tokyo"
    log.info("PASS | test_pre_model_before_model_hook")


def test_dynamic_prompt_wrap_model_call_hook():
    """wrap_model_call hook must inject 'prompt' into the request dict."""
    builder = DynamicPromptBuilder()
    captured = {}

    def fake_handler(req):
        captured.update(req)
        return "model_response"

    request = {"agent_type": "weather", "context": {}, "cleaned_input": "weather in Paris"}
    builder.wrap_model_call(request, fake_handler)

    assert "prompt" in captured
    assert "pack" in captured["prompt"].lower()
    log.info("PASS | test_dynamic_prompt_wrap_model_call_hook")


def test_post_model_after_model_hook():
    """after_model hook must write 'cleaned_response' into state."""
    mw = PostModelMiddleware()
    state = {"raw_response": "**Tokyo** is great! Visit http://example.com"}
    result = mw.after_model(state)
    assert "cleaned_response" in result
    assert "**" not in result["cleaned_response"]
    assert "http" not in result["cleaned_response"]
    log.info("PASS | test_post_model_after_model_hook")


def test_pipeline_runs_all_hooks_in_order():
    """MiddlewarePipeline must execute before_model, wrap_model_call, and after_model
    in the correct order and accumulate state changes across all three hooks."""
    pipeline = MiddlewarePipeline([
        PreModelMiddleware(),
        DynamicPromptBuilder(),
        PostModelMiddleware(),
    ])

    # Step 1 — before_model cleans input
    state = {"user_input": "  flights to  Tokyo  ", "conversation_history": []}
    state = pipeline.run_before_model(state)
    assert state["cleaned_input"] == "flights to Tokyo"
    assert state["metadata"]["detected_city"] == "Tokyo"

    # Step 2 — wrap_model_call injects prompt
    captured = {}

    def fake_model(req):
        captured.update(req)
        return "flight response"

    request = {
        "agent_type": "flight",
        "context": {},
        "cleaned_input": state["cleaned_input"],
    }
    pipeline.run_wrap_model_call(request, fake_model)
    assert "prompt" in captured
    assert "IATA" in captured["prompt"] or "JFK" in captured["prompt"]

    # Step 3 — after_model cleans output
    state["raw_response"] = "**Best flights** from JFK to NRT: http://flights.example.com"
    state = pipeline.run_after_model(state)
    assert "cleaned_response" in state
    assert "**" not in state["cleaned_response"]
    assert "http" not in state["cleaned_response"]
    log.info("PASS | test_pipeline_runs_all_hooks_in_order")
