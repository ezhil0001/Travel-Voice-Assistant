import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { Message, ToolEvent, VoiceIntent } from '../../models/message.model';
import { VoiceState } from '../../models/voice-state.model';

@Injectable({ providedIn: 'root' })
export class ChatStateService {

  private _messages         = new BehaviorSubject<Message[]>([]);
  private _activeToolEvents = new BehaviorSubject<ToolEvent[]>([]);
  private _voiceState       = new BehaviorSubject<VoiceState>({
    status:           'idle',
    silenceCountdown: 4,
    liveTranscript:   '',
    errorMessage:     '',
    isTtsSpeaking:    false,
  });

  messages$         = this._messages.asObservable();
  activeToolEvents$ = this._activeToolEvents.asObservable();
  voiceState$       = this._voiceState.asObservable();

  getMessages(): Message[] {
    return this._messages.getValue();
  }

  addMessage(msg: Message): void {
    this._messages.next([...this._messages.getValue(), msg]);
  }

  updateMessage(id: string, partial: Partial<Message>): void {
    const updated = this._messages.getValue().map(m =>
      m.id === id ? { ...m, ...partial } : m
    );
    this._messages.next(updated);
  }

  removeMessage(id: string): void {
    this._messages.next(this._messages.getValue().filter(m => m.id !== id));
  }

  addToolEvent(event: ToolEvent): void {
    this._activeToolEvents.next([...this._activeToolEvents.getValue(), event]);
  }

  clearActiveToolEvents(): void {
    this._activeToolEvents.next([]);
  }

  setVoiceState(patch: Partial<VoiceState>): void {
    this._voiceState.next({ ...this._voiceState.getValue(), ...patch });
  }

  resetVoiceState(): void {
    this._voiceState.next({
      status: 'idle', silenceCountdown: 4,
      liveTranscript: '', errorMessage: '', isTtsSpeaking: false,
    });
  }

  clearConversation(): void {
    this._messages.next([]);
    this._activeToolEvents.next([]);
    this.resetVoiceState();
  }

  generateId(): string {
    return 'id_' + Date.now() + '_' + Math.random().toString(36).substr(2, 5);
  }

  buildUserMessage(text: string): Message {
    return { id: this.generateId(), role: 'user', text, timestamp: new Date() };
  }

  buildLoadingMessage(): Message {
    return { id: this.generateId(), role: 'assistant', text: '', timestamp: new Date(), isLoading: true };
  }

  resolveMessage(id: string, text: string, intent: VoiceIntent | string, toolEvents: ToolEvent[]): void {
    this.updateMessage(id, {
      text, intent: intent as VoiceIntent, toolEvents, isLoading: false, timestamp: new Date(),
    });
  }
}
