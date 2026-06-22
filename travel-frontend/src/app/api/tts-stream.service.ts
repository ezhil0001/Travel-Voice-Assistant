// TtsStreamService — real-time streaming TTS client.
//
// Connects to the backend /tts/stream WebSocket, sends the assistant's
// response text, and delivers audio chunks as an Observable<TtsEvent> that
// the VoicePanel can pipe directly into an AudioContext without waiting
// for the full audio file to be synthesised.
//
// Lifecycle
// ─────────
//   1. Call connect()            — opens the WebSocket
//   2. Call speak(text, options) — sends start_tts, returns Observable<TtsEvent>
//   3. Audio chunks arrive as TtsAudioEvent — decode base64 and enqueue in AudioContext
//   4. Observable completes when TtsDoneEvent arrives
//   5. Call stop()               — sends stop_tts (early abort, e.g. skip button)
//   6. Call disconnect()         — closes the socket
//
// The speak() Observable never errors — provider failures arrive as TtsErrorEvent
// so the audio pipeline keeps running and the component can show a fallback message.

import { Injectable, OnDestroy } from '@angular/core';
import { Observable, Subject } from 'rxjs';
import { filter } from 'rxjs/operators';
import { environment } from '../../environments/environment';

// ── Event types ──────────────────────────────────────────────────────────────

export interface TtsReadyEvent        { type: 'tts_ready' }
export interface TtsAudioEvent        { type: 'tts_audio';        audio_base64: string; format: string }
export interface TtsDoneEvent         { type: 'tts_done' }
export interface TtsErrorEvent        { type: 'tts_error';        message: string }
export interface TtsStoppedEvent      { type: 'tts_stopped' }
export interface TtsDisconnectedEvent { type: 'tts_disconnected'; code: number; reason: string }

export type TtsEvent =
  | TtsReadyEvent
  | TtsAudioEvent
  | TtsDoneEvent
  | TtsErrorEvent
  | TtsStoppedEvent
  | TtsDisconnectedEvent;

export interface TtsSpeakOptions {
  languageCode?: string;  // BCP-47, e.g. 'en-IN' (default) or 'hi-IN'
  speaker?:      string;  // Sarvam speaker name, e.g. 'meera' (default)
}

// ── Service ───────────────────────────────────────────────────────────────────

@Injectable({ providedIn: 'root' })
export class TtsStreamService implements OnDestroy {

  private ws: WebSocket | null = null;
  private events$ = new Subject<TtsEvent>();

  /** All TTS events from the server. Never errors — completes on disconnect. */
  get events(): Observable<TtsEvent> {
    return this.events$.asObservable();
  }

  /** Open the WebSocket connection. Call once per session, before speak(). */
  connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }
    this.ws = new WebSocket(`${environment.wsBaseUrl}/tts/stream`);

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const data: TtsEvent = JSON.parse(event.data as string);
        this.events$.next(data);
      } catch {
        // Malformed frame — ignore
      }
    };

    this.ws.onerror = () => {
      this.events$.next({ type: 'tts_error', message: 'WebSocket connection error' });
    };

    this.ws.onclose = (evt: CloseEvent) => {
      this.events$.next({ type: 'tts_disconnected', code: evt.code, reason: evt.reason || '' });
    };
  }

  /**
   * Synthesise text and return an Observable that emits audio chunk events.
   *
   * The Observable completes when the server sends tts_done (synthesis finished)
   * or tts_stopped (early abort). It never throws — tts_error events are emitted
   * as values so callers handle them inline rather than in an error handler.
   *
   * Usage:
   *   this.tts.speak('Tokyo is 18°C today.').subscribe(evt => {
   *     if (evt.type === 'tts_audio') decodeAndEnqueue(evt.audio_base64);
   *   });
   */
  speak(text: string, options: TtsSpeakOptions = {}): Observable<TtsEvent> {
    this._sendJson({
      type:          'start_tts',
      text,
      language_code: options.languageCode ?? 'en-IN',
      speaker:       options.speaker      ?? 'meera',
    });

    // Return a filtered view that starts from tts_ready and ends at tts_done/stopped/error
    return new Observable<TtsEvent>(observer => {
      const sub = this.events$.subscribe(evt => {
        observer.next(evt);
        if (evt.type === 'tts_done' || evt.type === 'tts_stopped') {
          observer.complete();
          sub.unsubscribe();
        }
        if (evt.type === 'tts_error' || evt.type === 'tts_disconnected') {
          observer.complete();
          sub.unsubscribe();
        }
      });
      return () => sub.unsubscribe();
    });
  }

  /**
   * Decode a base64 audio chunk from a TtsAudioEvent into an ArrayBuffer.
   *
   * The caller enqueues this into an AudioContext.decodeAudioData() pipeline
   * for low-latency progressive playback.
   */
  decodeAudioChunk(base64: string): ArrayBuffer {
    const binary = atob(base64);
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }

  /** Abort ongoing synthesis — triggers tts_stopped event from the server. */
  stop(): void {
    this._sendJson({ type: 'stop_tts' });
  }

  /** Close the WebSocket connection entirely. */
  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'client-disconnect');
      this.ws = null;
    }
  }

  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  ngOnDestroy(): void {
    this.disconnect();
    this.events$.complete();
  }

  // ── private ─────────────────────────────────────────────────────────────────

  private _sendJson(payload: object): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.events$.next({ type: 'tts_error', message: 'WebSocket is not open' });
      return;
    }
    this.ws.send(JSON.stringify(payload));
  }
}
