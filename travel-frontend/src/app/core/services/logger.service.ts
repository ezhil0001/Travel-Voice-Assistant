import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class LoggerService {

  log(tag: string, data?: unknown): void {
    console.log(`[${tag}]`, data ?? '');
  }

  warn(tag: string, data?: unknown): void {
    console.warn(`[${tag}]`, data ?? '');
  }

  error(tag: string, err?: unknown): void {
    console.error(`[${tag} ERROR]`, err ?? '');
  }
}
