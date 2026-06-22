import { Injectable, OnDestroy } from '@angular/core';
import { Observable, Subject } from 'rxjs';
import { environment } from '../../environments/environment';

// ── Event types ──────────────────────────────────────────────────────────────

export interface SttReadyEvent        { type: 'stt_ready' }
export interface SttInterimEvent      { type: 'stt_interim';       transcript: string; is_final: false }
export interface SttFinalEvent        { type: 'stt_final';         transcript: string; is_final: true  }
export interface SttErrorEvent        { type: 'stt_error';         message: string }
export interface SttStoppedEvent      { type: 'stt_stopped' }
export interface SttDisconnectedEvent { type: 'stt_disconnected';  code: number; reason: string }

export type SttEvent =
  | SttReadyEvent
  | SttInterimEvent
  | SttFinalEvent
  | SttErrorEvent
  | SttStoppedEvent
  | SttDisconnectedEvent;

// ── Constants ─────────────────────────────────────────────────────────────────
// Maximum audio frames to queue while waiting for stt_ready.
// At ~93 ms / frame (4096 samples @ 44 100 Hz) this is ~18 s of audio —
// comfortably exceeds the backend's 10-second upstream CONNECT_TIMEOUT.
const MAX_AUDIO_QUEUE = 200;

// ── Service ───────────────────────────────────────────────────────────────────

@Injectable({ providedIn: 'root' })
export class SttStreamService implements OnDestroy {

  private ws: WebSocket | null = null;
  private events$ = new Subject<SttEvent>();

  // ── Session readiness tracking ────────────────────────────────────────────
  //
  //  The backend only starts forwarding audio to Sarvam/Deepgram AFTER it
  //  sends stt_ready (which arrives once it has opened the upstream WS).
  //  Any binary frame arriving before that receives an error response.
  //
  //  Race 1 — startSession() called before ws.onopen:
  //    connect() creates WS (readyState = CONNECTING).
  //    startSession() tries _sendJson() → ws.readyState !== OPEN → stt_error.
  //    Fix: store the pending start params and send them from ws.onopen.
  //
  //  Race 2 — audio arrives before stt_ready:
  //    WS opens, start_stt sent, backend begins connecting upstream (~100–500 ms).
  //    ScriptProcessorNode is already firing → audio arrives at the backend
  //    before upstream is ready → "STT not initialised" errors.
  //    Fix: queue every chunk until stt_ready is received, then flush.

  /** True once the backend has confirmed the upstream STT connection is open. */
  private _sttReady = false;

  /** PCM frames queued while waiting for stt_ready. Flushed on stt_ready. */
  private _audioQueue: ArrayBuffer[] = [];

  /** Saved start_stt params when startSession() is called before ws.onopen. */
  private _pendingStart: { sessionId: string; lang: string } | null = null;

  // ─────────────────────────────────────────────────────────────────────────

  /** Emits every STT event from the server. Never errors — completes on disconnect. */
  get events(): Observable<SttEvent> {
    return this.events$.asObservable();
  }

  /** Open the WebSocket connection. Must be called before startSession(). */
  connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return; // already connected — startSession() handles the new turn
    }

    this._resetSessionState();

    this.ws = new WebSocket(`${environment.wsBaseUrl}/voice/stream`);
    this.ws.binaryType = 'arraybuffer';

    // ── onopen: WS handshake complete ──────────────────────────────────────
    //  startSession() may have been called before the socket opened.
    //  _pendingStart captures those params so we can send start_stt now.
    this.ws.onopen = () => {
      if (this._pendingStart) {
        this._sendJson({
          type:          'start_stt',
          session_id:    this._pendingStart.sessionId,
          language_code: this._pendingStart.lang,
        });
        this._pendingStart = null;
      }
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const data: SttEvent = JSON.parse(event.data as string);

        // ── stt_ready: upstream STT provider is connected ─────────────────
        //  This is the gate-open signal. Set the flag and flush every queued
        //  audio chunk so the backend gets a continuous stream from t=0.
        if (data.type === 'stt_ready') {
          this._sttReady = true;
          this._flushQueue();
        }

        this.events$.next(data);
      } catch {
        // Malformed frame — ignore
      }
    };

    this.ws.onerror = () => {
      this.events$.next({ type: 'stt_error', message: 'WebSocket connection error' });
    };

    this.ws.onclose = (evt: CloseEvent) => {
      this._resetSessionState();
      this.events$.next({
        type:   'stt_disconnected',
        code:   evt.code,
        reason: evt.reason || '',
      });
    };
  }

  /**
   * Send a start_stt control frame to open the upstream STT session.
   *
   * Safe to call immediately after connect() — if the socket is still
   * CONNECTING the params are stored and sent from the onopen callback.
   */
  startSession(sessionId: string, languageCode: string = 'en-IN'): void {
    // Reset readiness for this new turn — old queued chunks are discarded.
    this._sttReady = false;
    this._audioQueue = [];
    this._pendingStart = null;

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      // WS already open (e.g. a continuous-conversation second turn).
      this._sendJson({ type: 'start_stt', session_id: sessionId, language_code: languageCode });
    } else {
      // WS is still CONNECTING — save for ws.onopen.
      this._pendingStart = { sessionId, lang: languageCode };
    }
  }

  /**
   * Send a raw PCM audio chunk to the backend.
   *
   * If stt_ready has not yet been received, the chunk is queued so no
   * audio is lost during the upstream connection setup phase.
   * The queue is bounded to MAX_AUDIO_QUEUE frames to cap memory usage.
   */
  sendAudioChunk(pcmData: ArrayBuffer): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    if (!this._sttReady) {
      // Backend is not ready yet — queue the chunk.
      if (this._audioQueue.length < MAX_AUDIO_QUEUE) {
        this._audioQueue.push(pcmData);
      }
      return;
    }

    // Back-pressure guard: skip if socket buffer is already large.
    if (this.ws.bufferedAmount > 500_000) return;
    this.ws.send(pcmData);
  }

  /** Send a stop_stt control frame to close the upstream STT session. */
  stopSession(): void {
    this._resetSessionState();
    this._sendJson({ type: 'stop_stt' });
  }

  /** Close the WebSocket connection entirely. */
  disconnect(): void {
    this._resetSessionState();
    if (this.ws) {
      this.ws.close(1000, 'client-disconnect');
      this.ws = null;
    }
  }

  /** Whether the WebSocket is currently open. */
  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /** Whether the upstream STT session is open and ready to accept audio. */
  get isSttReady(): boolean {
    return this._sttReady;
  }

  ngOnDestroy(): void {
    this.disconnect();
    this.events$.complete();
  }

  // ── private ──────────────────────────────────────────────────────────────

  /** Flush queued audio chunks now that the backend is ready. */
  private _flushQueue(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this._audioQueue = [];
      return;
    }
    for (const chunk of this._audioQueue) {
      if (this.ws.bufferedAmount <= 500_000) {
        this.ws.send(chunk);
      }
    }
    this._audioQueue = [];
  }

  /** Reset all per-session state without closing the socket. */
  private _resetSessionState(): void {
    this._sttReady     = false;
    this._audioQueue   = [];
    this._pendingStart = null;
  }

  private _sendJson(payload: object): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.events$.next({ type: 'stt_error', message: 'WebSocket is not open' });
      return;
    }
    this.ws.send(JSON.stringify(payload));
  }
}
