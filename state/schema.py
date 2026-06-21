from typing import TypedDict


# Shared state schema for the LangGraph StateGraph.
#
# Every node reads from and writes to this TypedDict — it is the single
# source of truth that flows through the entire graph. No node receives
# raw function arguments; they read from state and write back to state.
#
# Field ownership by node:
#   user_input, conversation_history  → set by the caller (run_graph)
#   cleaned_input                     → written by pre_middleware node
#   detected_intent                   → written by supervisor node
#   sub_agent_response                → written by each agent node
#   final_response                    → written by post_middleware node
#   error                             → written by any node on failure


class TravelState(TypedDict):
    user_input:           str    # Raw transcribed text from the STT layer
    conversation_history: list   # [{role: str, content: str}, ...] — last N turns
    cleaned_input:        str    # Sanitised input after PreModelMiddleware
    detected_intent:      str    # One of: weather/flight/attractions/currency/timezone/general
    sub_agent_response:   str    # Raw response from the routed sub-agent
    final_response:       str    # TTS-ready response after PostModelMiddleware
    error:                str    # Non-empty string if any node encountered a failure
