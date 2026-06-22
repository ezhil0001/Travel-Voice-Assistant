// Domain types shared across the chat, voice, and API layers.

export type MessageRole = 'user' | 'assistant';

export type VoiceIntent =
  | 'weather'
  | 'flight'
  | 'attractions'
  | 'currency'
  | 'timezone'
  | 'general';

// Represents a single tool call the AI made while processing a query.
export interface ToolEvent {
  id: string;
  toolName: string;        // e.g. "get_weather"
  label: string;           // e.g. "Fetching weather data"
  status: 'running' | 'success' | 'error';
  detail?: string;         // focused query sent to the tool
  errorMessage?: string;
  timestamp: Date;
  durationMs?: number;     // execution time in milliseconds
  source?: string;         // e.g. "OpenWeatherMap API"
}

// A single turn in the conversation.
export interface Message {
  id: string;
  role: MessageRole;
  text: string;
  timestamp: Date;
  intent?: VoiceIntent;
  intents?: VoiceIntent[];           // all detected intents for multi-intent queries
  agentResponses?: Record<string, string>; // per-intent optimised responses from the backend
  summaryResponse?: string;          // short conversational summary derived from optimised output
  toolEvents?: ToolEvent[];
  isLoading?: boolean;
  isLiveTranscript?: boolean;
}
