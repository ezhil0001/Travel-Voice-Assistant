// Tracks everything the voice panel needs to render and the chat window
// needs to know about (e.g. whether to disable the text input).
// Using a single VoiceState object rather than individual BehaviorSubjects
// avoids extra emissions and keeps patches atomic.

export type VoiceStatus =
  | 'idle'              // mic is off, ready to start
  | 'listening'         // MediaRecorder + SpeechRecognition are active
  | 'silence_countdown' // no speech detected for a few seconds, auto-send imminent
  | 'processing'        // audio sent, waiting for backend response
  | 'speaking'          // TTS audio is playing
  | 'error';            // something went wrong — errorMessage has details

export interface VoiceState {
  status: VoiceStatus;
  silenceCountdown: number;  // counts from 4 down to 0
  liveTranscript: string;    // interim Web Speech API result shown in the textbox
  errorMessage: string;
  isTtsSpeaking: boolean;    // separate flag so the Skip button can bind cleanly
}
