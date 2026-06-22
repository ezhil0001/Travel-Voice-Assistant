import { Component, NgZone, OnDestroy, OnInit } from '@angular/core';
import { Observable, Subscription } from 'rxjs';
import { take } from 'rxjs/operators';

import { VoiceApiService } from '../../../../api/voice-api.service';
import { SttStreamService, SttInterimEvent, SttFinalEvent } from '../../../../api/stt-stream.service';
import { ChatStateService } from '../../../../core/services/chat-state.service';
import { SessionService } from '../../../../core/services/session.service';
import { LoggerService } from '../../../../core/services/logger.service';
import { ToolEvent } from '../../../../models/message.model';
import { VoiceState, VoiceStatus } from '../../../../models/voice-state.model';

// ─── VAD configuration ────────────────────────────────────────────────────────
const VAD_RMS_THRESHOLD = 12;
const VAD_SILENCE_GRACE = 3000;   // 3 s of silence → submit
const VAD_POLL_MS       = 100;

// ─── PCM streaming configuration ─────────────────────────────────────────────
// ScriptProcessorNode captures Float32 audio from the microphone, downsamples
// to PCM_TARGET_RATE (16 kHz mono Int16) and streams it to the backend
// WebSocket STT endpoint (/voice/stream) in real time.
const PCM_TARGET_RATE = 16000;   // Sarvam / Deepgram expected sample rate
const PCM_BUFFER_SIZE = 4096;    // ScriptProcessorNode buffer (samples per call)

@Component({
  standalone: false,
  selector: 'app-voice-panel',
  templateUrl: './voice-panel.component.html',
  styleUrls: ['./voice-panel.component.scss'],
})
export class VoicePanelComponent implements OnInit, OnDestroy {

  voiceState$: Observable<VoiceState>;
  toolEvents$: Observable<ToolEvent[]>;

  /** True while the continuous conversation loop is running. */
  conversationActive = false;

  // ── MediaRecorder — collects the full audio blob for POST /voice/query ─────
  private mediaRecorder?: MediaRecorder;
  private audioChunks: Blob[] = [];

  // ── Audio pipeline — shared for VAD and PCM streaming to WebSocket STT ─────
  private audioCtx?: AudioContext;
  private analyser?: AnalyserNode;
  private vadBuffer?: Uint8Array;
  private scriptProcessor?: ScriptProcessorNode;

  // ── VAD state ──────────────────────────────────────────────────────────────
  private vadPollTimer?: any;
  private silenceStart?: number;
  private hasSpeechBegun = false;

  // ── TTS ────────────────────────────────────────────────────────────────────
  private currentAudio?: HTMLAudioElement;

  // ── Transcript accumulator ─────────────────────────────────────────────────
  // Sarvam v3 emits one stt_final per spoken segment. We accumulate them so
  // the live user message bubble grows across the whole turn.
  //
  //   _committedTranscript — space-joined finals confirmed by END_SPEECH
  //   currentTranscript    — final value snapshotted by stopAndSend()
  //   _liveUserMsgId       — ID of the user bubble created on first speech
  //
  // Lifecycle:
  //   stt_interim("…")  → create bubble (if not yet) with "…", mark isLiveTranscript
  //   stt_final("Hello") → append to committed, update bubble text
  //   stopAndSend()      → mark isLiveTranscript=false, submit audio
  private _committedTranscript = '';
  private currentTranscript    = '';
  private _liveUserMsgId: string | null = null;

  // ── HTTP + WebSocket subscriptions ────────────────────────────────────────
  private pendingRequest?: Subscription;
  private pendingAssistantMsgId?: string;
  private sttSub?: Subscription;

  constructor(
    private voiceApi: VoiceApiService,
    private sttStream: SttStreamService,
    private chatState: ChatStateService,
    private session: SessionService,
    private logger: LoggerService,
    private zone: NgZone,
  ) {
    this.voiceState$ = this.chatState.voiceState$;
    this.toolEvents$ = this.chatState.activeToolEvents$;
  }

  ngOnInit(): void {
    // ── Subscribe to WebSocket STT events ────────────────────────────────────
    //  This is the AUTHORITATIVE real-time transcript source.
    //  stt_interim fires on every partial word as you speak — exactly what
    //  needs to appear live in the textarea. Browser SpeechRecognition is
    //  NOT used because it requires a synchronous user-gesture context and
    //  silently fails after any async gap (getUserMedia permission dialog).
    this.sttSub = this.sttStream.events.subscribe(event => {
      if (event.type === 'stt_interim') {
        // "…" = START_SPEECH signal → create the live bubble immediately if not yet present.
        // Other interims (Deepgram word-by-word) append to the committed text.
        const displayText = event.transcript === '…'
          ? (this._committedTranscript ? this._committedTranscript + ' …' : '…')
          : (this._committedTranscript ? this._committedTranscript + ' ' + event.transcript : event.transcript);

        this.zone.run(() => {
          if (!this._liveUserMsgId) {
            // First speech detected — create the user message bubble live in the chat
            this._liveUserMsgId = this.chatState.generateId();
            this.chatState.addMessage({
              id: this._liveUserMsgId,
              role: 'user',
              text: displayText,
              timestamp: new Date(),
              isLiveTranscript: true,
            });
          } else {
            this.chatState.updateMessage(this._liveUserMsgId, { text: displayText });
          }
          // Keep input area clear — transcript lives in the bubble now
          this.chatState.setVoiceState({ liveTranscript: '' });
        });

      } else if (event.type === 'stt_final') {
        // Commit this segment and update the live bubble
        this._committedTranscript = this._committedTranscript
          ? this._committedTranscript + ' ' + event.transcript
          : event.transcript;
        this.currentTranscript = this._committedTranscript;

        this.zone.run(() => {
          if (!this._liveUserMsgId) {
            // Edge case: final arrived before any interim (e.g. Deepgram)
            this._liveUserMsgId = this.chatState.generateId();
            this.chatState.addMessage({
              id: this._liveUserMsgId,
              role: 'user',
              text: this._committedTranscript,
              timestamp: new Date(),
              isLiveTranscript: true,
            });
          } else {
            this.chatState.updateMessage(this._liveUserMsgId, {
              text: this._committedTranscript,
            });
          }
          this.chatState.setVoiceState({ liveTranscript: '' });
        });

      } else if (event.type === 'stt_error') {
        this.logger.warn('VoicePanel', `STT error: ${event.message}`);
        this.zone.run(() =>
          this.chatState.setVoiceState({ status: 'error', errorMessage: event.message })
        );
      } else if (event.type === 'stt_ready') {
        this.logger.log('VoicePanel', 'WebSocket STT backend ready');
      }
    });
  }

  // ─── Mic button ───────────────────────────────────────────────────────────

  onMicClick(): void {
    let currentStatus!: VoiceStatus;
    this.chatState.voiceState$.pipe(take(1)).subscribe(s => (currentStatus = s.status));

    if (currentStatus === 'listening' || currentStatus === 'silence_countdown') {
      this.stopAndSend();
    } else if (currentStatus === 'idle' || currentStatus === 'error') {
      this.conversationActive = true;
      this.startListening();
    }
  }

  // ─── End session ──────────────────────────────────────────────────────────

  stopConversation(): void {
    this.conversationActive = false;

    if (this.pendingRequest) {
      this.pendingRequest.unsubscribe();
      this.pendingRequest = undefined;
    }
    if (this.pendingAssistantMsgId) {
      this.chatState.removeMessage(this.pendingAssistantMsgId);
      this.pendingAssistantMsgId = undefined;
    }

    this.stopVAD();
    this.stopPcmStream();
    this.sttStream.stopSession();

    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.onstop = () =>
        this.mediaRecorder?.stream.getTracks().forEach(t => t.stop());
      this.mediaRecorder.stop();
    }
    if (this.currentAudio) {
      this.currentAudio.onended = null;
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
    }
    this.chatState.setVoiceState({
      status: 'idle', isTtsSpeaking: false, liveTranscript: '', errorMessage: '',
    });
    this.logger.log('VoicePanel', 'Conversation session ended by user');
  }

  // ─── Start listening ──────────────────────────────────────────────────────
  //
  //  Audio pipeline per turn:
  //
  //    getUserMedia()
  //      └─ AudioContext
  //           ├─ AnalyserNode          (VAD — silence detection)
  //           └─ ScriptProcessorNode   (PCM streaming → WebSocket STT)
  //      └─ MediaRecorder             (full audio blob → POST /voice/query)

  private async startListening(): Promise<void> {
    this.audioChunks          = [];
    this.hasSpeechBegun       = false;
    this.silenceStart         = undefined;
    this.currentTranscript    = '';
    this._committedTranscript = '';
    this._liveUserMsgId       = null;

    // ── Open WebSocket + send start_stt BEFORE getUserMedia ─────────────────
    //  This lets the backend handshake with Sarvam/Deepgram in parallel while
    //  the browser shows the microphone-permission dialog.
    //  Timeline with this change:
    //    t=0   connect() + startSession() called → WS handshake begins
    //    t≈50  ws.onopen fires → start_stt sent immediately
    //    t≈50–500 ms  backend opens upstream STT provider connection
    //    t≈X   permission granted → getUserMedia resolves → AudioContext starts
    //    t≈X   stt_ready received → _flushQueue() drains any buffered chunks
    //  The audio queue in SttStreamService absorbs every chunk produced while
    //  _sttReady is false, so zero audio is lost.
    this.sttStream.connect();
    this.sttStream.startSession(this.session.getSessionId(), 'en-IN');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // ── MediaRecorder: collect full audio for POST /voice/query ───────────
      this.mediaRecorder = new MediaRecorder(stream);
      this.mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) this.audioChunks.push(e.data);
      };
      this.mediaRecorder.start(100);

      // ── AudioContext: shared source for VAD and PCM streaming ─────────────
      this.audioCtx = new AudioContext();
      const source  = this.audioCtx.createMediaStreamSource(stream);

      // AnalyserNode — VAD
      this.analyser          = this.audioCtx.createAnalyser();
      this.analyser.fftSize  = 256;
      this.vadBuffer         = new Uint8Array(this.analyser.fftSize);
      source.connect(this.analyser);

      // ScriptProcessorNode — downsample Float32 → Int16 PCM, send to WebSocket.
      // sendAudioChunk() queues internally until stt_ready; no guard needed here.
      this.scriptProcessor = this.audioCtx.createScriptProcessor(PCM_BUFFER_SIZE, 1, 1);
      source.connect(this.scriptProcessor);
      this.scriptProcessor.connect(this.audioCtx.destination);

      const nativeRate = this.audioCtx.sampleRate;
      this.scriptProcessor.onaudioprocess = (e: AudioProcessingEvent) => {
        const pcm = this.downsampleToPcm16(e.inputBuffer.getChannelData(0), nativeRate, PCM_TARGET_RATE);
        this.sttStream.sendAudioChunk(pcm.buffer);
      };

      this.chatState.setVoiceState({
        status: 'listening', liveTranscript: '', errorMessage: '', silenceCountdown: 0,
      });
      this.chatState.clearActiveToolEvents();

      // ── VAD polling — outside zone to avoid 100 ms CD thrash ─────────────
      this.zone.runOutsideAngular(() => {
        this.vadPollTimer = setInterval(() => this.vadTick(), VAD_POLL_MS);
      });

      this.logger.log('VoicePanel', 'Listening started (WebSocket STT + VAD active)');

    } catch (err) {
      this.conversationActive = false;
      this.sttStream.stopSession();
      this.chatState.setVoiceState({
        status: 'error',
        errorMessage: 'Microphone access denied. Please allow microphone permissions.',
      });
      this.logger.error('VoicePanel', err as Error);
    }
  }

  // ─── PCM downsampler ──────────────────────────────────────────────────────
  //  Float32 samples at the browser's native rate (44100 / 48000 Hz)
  //  → Int16 PCM at 16000 Hz (what Sarvam and Deepgram expect).

  private downsampleToPcm16(input: Float32Array, fromRate: number, toRate: number): Int16Array {
    const ratio  = fromRate / toRate;
    const length = Math.floor(input.length / ratio);
    const output = new Int16Array(length);
    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
      output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    }
    return output;
  }

  // ─── Web Audio VAD ────────────────────────────────────────────────────────

  private vadTick(): void {
    if (!this.analyser || !this.vadBuffer) return;

    this.analyser.getByteTimeDomainData(this.vadBuffer);
    let sumSq = 0;
    for (let i = 0; i < this.vadBuffer.length; i++) {
      const v = (this.vadBuffer[i] - 128) / 128;
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / this.vadBuffer.length) * 255;

    if (rms >= VAD_RMS_THRESHOLD) {
      this.hasSpeechBegun = true;
      this.silenceStart   = undefined;
      this.zone.run(() =>
        this.chatState.setVoiceState({ status: 'listening', silenceCountdown: 0 })
      );
    } else {
      if (!this.hasSpeechBegun) return;
      const now = Date.now();
      if (!this.silenceStart) this.silenceStart = now;
      const elapsed   = now - this.silenceStart;
      const remaining = Math.max(0, (VAD_SILENCE_GRACE - elapsed) / 1000);
      this.zone.run(() =>
        this.chatState.setVoiceState({
          status: 'silence_countdown', silenceCountdown: parseFloat(remaining.toFixed(1)),
        })
      );
      if (elapsed >= VAD_SILENCE_GRACE) {
        this.logger.log('VoicePanel', `VAD: ${(elapsed / 1000).toFixed(2)}s silence → sending`);
        this.zone.run(() => this.stopAndSend());
      }
    }
  }

  private stopVAD(): void {
    clearInterval(this.vadPollTimer);
    this.vadPollTimer   = undefined;
    this.analyser       = undefined;
    this.vadBuffer      = undefined;
    this.silenceStart   = undefined;
    this.hasSpeechBegun = false;
  }

  private stopPcmStream(): void {
    if (this.scriptProcessor) {
      this.scriptProcessor.disconnect();
      (this.scriptProcessor as any).onaudioprocess = null;
      this.scriptProcessor = undefined;
    }
    if (this.audioCtx) {
      this.audioCtx.close().catch(() => { /* ignore */ });
      this.audioCtx = undefined;
    }
  }

  // ─── Stop recorder and send ───────────────────────────────────────────────

  private stopAndSend(): void {
    this.stopVAD();
    this.stopPcmStream();
    this.sttStream.stopSession();

    // Seal the live bubble — remove the pulsing cursor, show final committed text
    if (this._liveUserMsgId) {
      this.chatState.updateMessage(this._liveUserMsgId, {
        text: this.currentTranscript || '…',
        isLiveTranscript: false,
      });
    }

    if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') return;

    this.mediaRecorder.onstop = () => {
      const finalTranscript = this.currentTranscript;
      const mimeType = this.mediaRecorder?.mimeType || 'audio/webm';
      const blob = new Blob(this.audioChunks, { type: mimeType });
      this.mediaRecorder?.stream.getTracks().forEach(t => t.stop());
      this.sendAudio(blob, finalTranscript);
    };
    this.mediaRecorder.stop();
    this.chatState.setVoiceState({ status: 'processing' });
  }

  // ─── Send to backend ──────────────────────────────────────────────────────

  private sendAudio(blob: Blob, sttTranscript: string): void {
    const displayText = sttTranscript.trim() || '…';

    // Reuse the live bubble that was updated during STT streaming.
    // If somehow no bubble was created (e.g. silence with no speech detected),
    // create a new one now.
    const userMsgId = this._liveUserMsgId ?? this.chatState.generateId();
    if (!this._liveUserMsgId) {
      this.chatState.addMessage({ id: userMsgId, role: 'user', text: displayText, timestamp: new Date() });
    } else {
      // Ensure isLiveTranscript is false (cursor gone) and text is final
      this.chatState.updateMessage(userMsgId, { text: displayText, isLiveTranscript: false });
    }
    this._liveUserMsgId = null;

    const assistantMsgId = this.chatState.generateId();
    this.chatState.addMessage({ id: assistantMsgId, role: 'assistant', text: '', timestamp: new Date(), isLoading: true });
    this.pendingAssistantMsgId = assistantMsgId;

    this.pendingRequest = this.voiceApi.sendAudio(blob, this.session.getSessionId()).subscribe({
      next: res => {
        this.pendingRequest        = undefined;
        this.pendingAssistantMsgId = undefined;
        if (!this.conversationActive) return;

        // res.transcript is Whisper's authoritative server-side transcription.
        // Use it if the WebSocket STT was empty (e.g. WS connection failed).
        const finalText = (res.transcript || '').trim() || sttTranscript.trim() || '(voice message)';
        this.chatState.updateMessage(userMsgId, { text: finalText });

        const toolEvents: ToolEvent[] = (res.tool_events || []).map((t: any) => ({
          id: this.chatState.generateId(), toolName: t.tool_name, label: t.label || t.tool_name,
          status: t.status, detail: t.detail, errorMessage: t.error,
          timestamp: new Date(), durationMs: t.duration_ms, source: t.source,
        }));
        this.chatState.updateMessage(assistantMsgId, {
          text:            res.response,
          intent:          res.intent as any,
          intents:         (res.intents || [res.intent]).filter(Boolean) as any,
          agentResponses:  res.agent_responses || {},
          summaryResponse: res.summary_response || res.response,
          toolEvents,
          isLoading:       false,
        });
        toolEvents.forEach(e => this.chatState.addToolEvent(e));
        this.playAudio(res.audioBlob);
        this.logger.log('VoicePanel', `Response received — transcript: "${finalText}"`);
      },
      error: err => {
        this.pendingRequest        = undefined;
        this.pendingAssistantMsgId = undefined;
        this.conversationActive    = false;
        this.chatState.updateMessage(assistantMsgId, {
          text: err.message || 'Something went wrong.', isLoading: false,
          toolEvents: [{
            id: this.chatState.generateId(), toolName: 'voice_api', label: 'Voice API call',
            status: 'error', errorMessage: err.message, timestamp: new Date(),
          }],
        });
        this.chatState.setVoiceState({ status: 'error', errorMessage: err.message });
      },
    });
  }

  // ─── TTS playback ─────────────────────────────────────────────────────────

  private playAudio(audioBlob: Blob): void {
    const url = URL.createObjectURL(audioBlob);
    this.currentAudio = new Audio(url);
    this.chatState.setVoiceState({ status: 'speaking', isTtsSpeaking: true });

    this.currentAudio.onended = () => {
      URL.revokeObjectURL(url);
      if (!this.conversationActive) {
        this.chatState.setVoiceState({ status: 'idle', isTtsSpeaking: false, liveTranscript: '' });
        return;
      }
      this.chatState.setVoiceState({ isTtsSpeaking: false, liveTranscript: '' });
      this.logger.log('VoicePanel', 'TTS finished — auto-resuming STT');
      this.startListening();
    };

    this.currentAudio.play().catch(err => {
      this.logger.error('VoicePanel TTS play', err);
      this.chatState.setVoiceState({ status: 'idle', isTtsSpeaking: false });
      if (this.conversationActive) this.startListening();
    });
  }

  // ─── Skip TTS ─────────────────────────────────────────────────────────────

  skipTts(): void {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
    }
    this.chatState.setVoiceState({ isTtsSpeaking: false, liveTranscript: '' });
    if (this.conversationActive) {
      this.logger.log('VoicePanel', 'TTS skipped — resuming STT');
      this.startListening();
    } else {
      this.chatState.setVoiceState({ status: 'idle' });
    }
  }

  ngOnDestroy(): void {
    this.conversationActive = false;
    if (this.pendingRequest) { this.pendingRequest.unsubscribe(); this.pendingRequest = undefined; }
    if (this.sttSub)         { this.sttSub.unsubscribe();         this.sttSub = undefined; }
    this.stopVAD();
    this.stopPcmStream();
    this.sttStream.stopSession();
    this.sttStream.disconnect();
    if (this.currentAudio) { this.currentAudio.onended = null; this.currentAudio.pause(); }
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.onstop = () => this.mediaRecorder?.stream.getTracks().forEach(t => t.stop());
      this.mediaRecorder.stop();
    }
  }
}
