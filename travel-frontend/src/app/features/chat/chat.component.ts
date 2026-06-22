import { Component, OnInit } from '@angular/core';
import { ChatStateService } from '../../core/services/chat-state.service';
import { VoiceApiService } from '../../api/voice-api.service';

const WELCOME_TEXT =
  "Hello! I'm your travel planning assistant. You can type or use the mic on the right. Ask me about weather, flights, attractions, currency, or time zones!";

@Component({
  standalone: false,
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss'],
})
export class ChatComponent implements OnInit {
  constructor(
    private chatState: ChatStateService,
    private voiceApi: VoiceApiService,
  ) {}

  ngOnInit(): void {
    // Seed welcome message
    this.chatState.addMessage({
      id: this.chatState.generateId(),
      role: 'assistant',
      text: WELCOME_TEXT,
      timestamp: new Date(),
      intent: 'general',
    });

    // Auto-play the welcome greeting via TTS so the user hears it on load.
    // Uses a small POST-only synthesize endpoint — no mic, no graph run.
    // The play() call is wrapped in a short timeout to ensure the browser
    // audio context is ready (some browsers block autoplay before any user
    // interaction; we respect that by swallowing the NotAllowedError silently
    // — the text message is already visible either way).
    this.voiceApi.synthesizeWelcome(WELCOME_TEXT).subscribe({
      next: (audioBlob) => {
        const url   = URL.createObjectURL(audioBlob);
        const audio = new Audio(url);

        audio.onended = () => {
          URL.revokeObjectURL(url);
          this.chatState.setVoiceState({ status: 'idle', isTtsSpeaking: false });
        };

        // Small delay so the page has fully rendered before audio starts
        setTimeout(() => {
          audio.play()
            .then(() => {
              // Audio is actually playing — tell the right panel
              this.chatState.setVoiceState({ status: 'speaking', isTtsSpeaking: false });
            })
            .catch(() => {
              // Autoplay blocked by browser policy — silently ignore;
              // the welcome text is already displayed in the chat.
              URL.revokeObjectURL(url);
            });
        }, 500);
      },
      error: () => {
        // TTS synthesize failed (backend down, etc.) — welcome text is still shown.
      },
    });
  }
}

