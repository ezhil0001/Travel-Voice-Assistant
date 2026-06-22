import { Component, EventEmitter, Input, Output } from '@angular/core';
import { VoiceStatus } from '../../../models/voice-state.model';

@Component({
  standalone: false,
  selector: 'app-mic-button',
  templateUrl: './mic-button.component.html',
  styleUrls: ['./mic-button.component.scss'],
})
export class MicButtonComponent {
  @Input() status: VoiceStatus = 'idle';
  @Output() micClick = new EventEmitter<void>();

  get label(): string {
    switch (this.status) {
      case 'listening':         return 'Listening...';
      case 'silence_countdown': return 'Sending soon...';
      case 'processing':        return 'Processing...';
      case 'speaking':          return 'Speaking...';
      default:                  return 'Tap to Speak';
    }
  }

  get icon(): string {
    if (this.status === 'processing') return '⏳';
    if (this.status === 'speaking')   return '🔊';
    if (this.status === 'listening' || this.status === 'silence_countdown') return '⏹';
    return '🎙️';
  }

  get isActive(): boolean {
    return this.status === 'listening' || this.status === 'silence_countdown';
  }
}
