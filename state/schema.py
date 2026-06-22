from typing import TypedDict


# Shared state schema for the LangGraph StateGraph.
#
# Every node reads from and writes to this TypedDict — it is the single
# source of truth that flows through the entire graph.
#
# Field ownership by node:
#   user_input, conversation_history  → set by the caller (run_graph_full)
#   cleaned_input                     → written by pre_middleware node
#   detected_intents                  → written by supervisor node (list)
#   agent_responses                   → written by run_agents node (dict)
#   tool_events                       → written by run_agents node (list)
#   final_response                    → written by post_middleware node
#   error                             → written by any node on failure


class TravelState(TypedDict):
    user_input:           str         # Raw transcribed text from the STT layer
    conversation_history: list        # [{role: str, content: str}, ...] — last N turns
    cleaned_input:        str         # Sanitised input after PreModelMiddleware
    detected_intents:     list        # All detected intents e.g. ["weather","currency"]
    agent_responses:      dict        # {intent: response_str} from each agent
    tool_events:          list        # [{tool_name, label, status, detail}] for frontend
    final_response:       str         # TTS-ready merged response after PostModelMiddleware
    summary_response:     str         # Short conversational summary for the Summary View
    error:                str         # Non-empty string if any node encountered a failure
