import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';

import { environment } from '../../environments/environment';

// Shape of a single tool-call entry returned alongside the assistant response.
// The backend does not yet surface per-tool telemetry, so tool_events is always
// an empty array today — but the type is defined here so the message bubble
// can bind to it without a service change when the backend adds it.
export interface TextToolEvent {
  tool_name: string;
  label: string;
  status: string;
  detail?: string;
  error?: string;
}

export interface TextQueryResponse {
  response: string;
  intent: string;
  intents: string[];
  tool_events: TextToolEvent[];
  agent_responses: Record<string, string>;
  summary_response: string;
}

@Injectable({ providedIn: 'root' })
export class TextApiService {

  constructor(private http: HttpClient) {}

  /**
   * Sends a typed message to the backend text pipeline.
   *
   * The session ID is passed explicitly here rather than relying solely on
   * the HTTP interceptor so the component can see which session was used —
   * useful during multi-tab debugging.
   *
   * The map() ensures tool_events is always an array even when the backend
   * omits the field, preventing *ngFor from throwing on undefined.
   */
  sendText(text: string, sessionId: string): Observable<TextQueryResponse> {
    return this.http
      .post<any>(`${environment.apiBaseUrl}/text/query`, {
        text,
        session_id: sessionId,
      })
      .pipe(
        map(res => ({
          response:         res.response         || '',
          intent:           res.intent           || 'general',
          intents:          res.intents          || [res.intent || 'general'],
          tool_events:      res.tool_events      || [],
          agent_responses:  res.agent_responses  || {},
          summary_response: res.summary_response || res.response || '',
        }))
      );
  }
}
