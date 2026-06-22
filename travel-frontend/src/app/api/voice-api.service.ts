import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { environment } from '../../environments/environment';

export interface VoiceQueryResponse {
  audioBlob:        Blob;
  transcript:       string;
  response:         string;
  intent:           string;
  intents:          string[];
  tool_events:      any[];
  agent_responses:  Record<string, string>;
  summary_response: string;
}

@Injectable({ providedIn: 'root' })
export class VoiceApiService {

  constructor(private http: HttpClient) {}

  /**
   * Posts a recorded audio blob to the voice pipeline.
   *
   * The backend now returns JSON (not raw audio bytes) so the client can
   * populate the message bubble with text, intent badge, and tool activity
   * alongside playing the audio — all from a single HTTP round-trip.
   *
   * The X-Session-Id header is duplicated here (the interceptor also injects it)
   * because FormData requests sometimes strip custom headers in certain CORS
   * preflight configurations. Sending it explicitly guarantees it arrives.
   */
  sendAudio(audioBlob: Blob, sessionId: string): Observable<VoiceQueryResponse> {
    const formData = new FormData();
    // MediaRecorder in Chrome/Firefox always produces audio/webm regardless of
    // what MIME type the code requested.  Naming it .wav while it is actually
    // WebM makes Sarvam's REST endpoint fail with a 400.  Detect the real type
    // and send a consistent filename so the server-side magic-byte check works.
    const isWebm = audioBlob.type.includes('webm') || audioBlob.type === '';
    const filename = isWebm ? 'recording.webm' : 'recording.wav';
    formData.append('audio_file', audioBlob, filename);

    return this.http
      .post<any>(
        `${environment.apiBaseUrl}/voice/query`,
        formData,
        { headers: { 'X-Session-Id': sessionId } }
      )
      .pipe(
        map(res => ({
          audioBlob:        this.base64ToBlob(res.audio_base64, 'audio/wav'),
          transcript:       res.transcript       || '',
          response:         res.response         || '',
          intent:           res.intent           || 'general',
          intents:          res.intents          || [res.intent || 'general'],
          tool_events:      res.tool_events      || [],
          agent_responses:  res.agent_responses  || {},
          summary_response: res.summary_response || res.response || '',
        }))
      );
  }

  /**
   * Calls POST /tts/synthesize to convert plain text to speech without
   * going through the full STT → LangGraph → TTS pipeline.
   * Used exclusively for the auto-played welcome message on page load.
   */
  synthesizeWelcome(text: string): Observable<Blob> {
    return this.http
      .post<{ audio_base64: string }>(
        `${environment.apiBaseUrl}/tts/synthesize`,
        { text }
      )
      .pipe(
        map(res => this.base64ToBlob(res.audio_base64, 'audio/wav'))
      );
  }

  /**
   * Decodes a base64 string into a Blob the HTMLAudioElement can play.
   *
   * Using Uint8Array avoids the deprecated btoa/atob string-length limit
   * and handles binary data correctly regardless of audio duration.
   */
  private base64ToBlob(base64: string, mimeType: string): Blob {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length)
      .fill(0)
      .map((_, i) => byteCharacters.charCodeAt(i));
    return new Blob([new Uint8Array(byteNumbers)], { type: mimeType });
  }
}
