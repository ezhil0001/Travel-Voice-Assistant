import {
  AfterViewChecked, Component, ElementRef,
  NgZone, OnInit, OnDestroy, ViewChild,
} from '@angular/core';
import { Observable, Subscription } from 'rxjs';
import { TextApiService } from '../../../../api/text-api.service';
import { ChatStateService } from '../../../../core/services/chat-state.service';
import { SessionService } from '../../../../core/services/session.service';
import { LoggerService } from '../../../../core/services/logger.service';
import { Message, ToolEvent } from '../../../../models/message.model';
import { VoiceState } from '../../../../models/voice-state.model';

@Component({
  standalone: false,
  selector: 'app-chat-window',
  templateUrl: './chat-window.component.html',
  styleUrls: ['./chat-window.component.scss'],
})
export class ChatWindowComponent implements AfterViewChecked, OnInit, OnDestroy {
  @ViewChild('messagesEnd') messagesEnd!: ElementRef;

  messages$:   Observable<Message[]>;
  voiceState$: Observable<VoiceState>;

  /** Holds manual keyboard input. During voice turns, kept in sync with
   *  liveTranscript so sendText() can read the spoken text if the user
   *  presses Enter before the VAD fires. */
  inputText    = '';
  isSubmitting = false;

  private voiceSub?: Subscription;

  constructor(
    private chatState: ChatStateService,
    private textApi:   TextApiService,
    private session:   SessionService,
    private logger:    LoggerService,
    private zone:      NgZone,
  ) {
    this.messages$   = this.chatState.messages$;
    this.voiceState$ = this.chatState.voiceState$;
  }

  ngOnInit(): void {
    // The live transcript now renders inside the user message bubble directly.
    // The textarea stays blank during voice turns — only reset inputText when
    // the session ends so the text box is clean for the next typed message.
    this.voiceSub = this.voiceState$.subscribe(state => {
      if (state.status === 'idle' || state.status === 'error') {
        this.inputText = '';
      }
    });
  }

  ngAfterViewChecked(): void {
    this.scrollToBottom();
  }

  ngOnDestroy(): void {
    this.voiceSub?.unsubscribe();
  }

  trackById(_: number, msg: Message): string {
    return msg.id;
  }

  sendText(): void {
    const text = this.inputText.trim();
    if (!text || this.isSubmitting) return;

    const userMsgId = this.chatState.generateId();
    this.chatState.addMessage({ id: userMsgId, role: 'user', text, timestamp: new Date() });
    this.inputText = '';
    this.isSubmitting = true;
    this.chatState.clearActiveToolEvents();

    const loadingId = this.chatState.generateId();
    this.chatState.addMessage({
      id: loadingId, role: 'assistant', text: '',
      timestamp: new Date(), isLoading: true,
    });

    this.textApi.sendText(text, this.session.getSessionId()).subscribe({
      next: res => {
        const toolEvents: ToolEvent[] = (res.tool_events || []).map((t: any) => ({
          id: this.chatState.generateId(),
          toolName: t.tool_name, label: t.label || t.tool_name,
          status: t.status, detail: t.detail, errorMessage: t.error, timestamp: new Date(),
        }));
        this.chatState.updateMessage(loadingId, {
          text: res.response, intent: res.intent as any, toolEvents, isLoading: false,
        });
        toolEvents.forEach(e => this.chatState.addToolEvent(e));
        this.isSubmitting = false;
        this.logger.log('ChatWindow', 'Text response received');
      },
      error: err => {
        this.chatState.updateMessage(loadingId, {
          text: err.message || 'Server error. Please try again.',
          isLoading: false,
          toolEvents: [{
            id: this.chatState.generateId(),
            toolName: 'text_api', label: 'Text query',
            status: 'error', errorMessage: err.message, timestamp: new Date(),
          }],
        });
        this.isSubmitting = false;
      },
    });
  }

  onKeydown(e: KeyboardEvent): void {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.sendText(); }
  }

  /** Called on every (input) event from the textarea.
   *  During an active voice turn we discard user keystrokes so the STT
   *  transcript always owns the box. */
  onTextInput(event: Event): void {
    const el = event.target as HTMLTextAreaElement;
    if (this.isVoiceListening()) {
      // Revert any character the user managed to type — the box belongs to STT
      el.value = this.inputText;
      return;
    }
    this.inputText = el.value;
  }

  /** Returns true while STT is actively transcribing (listening / countdown). */
  isVoiceListening(): boolean {
    let s = '';
    this.chatState.voiceState$.subscribe(st => (s = st.status)).unsubscribe();
    return s === 'listening' || s === 'silence_countdown';
  }

  /** Returns true for ALL voice states where the send button should be locked. */
  isVoiceActive(state?: VoiceState | null): boolean {
    if (!state) return false;
    return ['listening', 'silence_countdown', 'processing', 'speaking'].includes(state.status);
  }

  private scrollToBottom(): void {
    try { this.messagesEnd.nativeElement.scrollIntoView({ behavior: 'smooth' }); }
    catch { /* not yet rendered */ }
  }
}
