import { Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class SessionService {

  /**
   * Return the current session ID, creating and persisting one if none exists.
   * Called on every HTTP request by the interceptor so the header is always current.
   */
  getSessionId(): string {
    let id = localStorage.getItem(environment.sessionKey);
    if (!id) {
      id = this._generateId();
      localStorage.setItem(environment.sessionKey, id);
    }
    return id;
  }

  /**
   * Discard the current session and return a fresh ID.
   * The backend will start a new conversation history for the new ID automatically.
   */
  resetSession(): string {
    localStorage.removeItem(environment.sessionKey);
    return this.getSessionId();
  }

  private _generateId(): string {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return crypto.randomUUID();
    }
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
    });
  }
}
