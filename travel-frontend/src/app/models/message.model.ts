// Domain types shared across the chat, voice, and API layers.
// Keeping them in one file means any component can import exactly
// what it needs without pulling in service logic.

export type MessageRole = 'user' | 'assistant';

export type VoiceIntent =
  | 'weather'
  | 'flight'
  | 'attractions'
  | 'currency'
  | 'timezone'
  | 'general';

// Represents a single tool call the AI made while processing a query.
// The UI renders these as collapsible panels under each assistant message.
export interface ToolEvent {
  id: string;
  toolName: string;       // e.g. "get_weather"
  label: string;          // human-readable description, e.g. "Fetching weather data"
  status: 'running' | 'success' | 'error';
  detail?: string;        // tool input summary, e.g. "city=Tokyo, units=metric"
  errorMessage?: string;  // only present when status === 'error'
  timestamp: Date;
}

// A single turn in the conversation — either a user query or an assistant reply.
export interface Message {
  id: string;
  role: MessageRole;
  text: string;
  timestamp: Date;
  intent?: VoiceIntent;      // set on assistant messages so the badge renders
  toolEvents?: ToolEvent[];  // populated once the backend responds
  isLoading?: boolean;       // true while waiting for the backend response
  isLiveTranscript?: boolean; // true while STT is streaming — shows live cursor in bubble
}
