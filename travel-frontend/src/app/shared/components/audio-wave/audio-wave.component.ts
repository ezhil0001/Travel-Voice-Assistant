import { Component, Input } from '@angular/core';

@Component({
  standalone: false,
  selector: 'app-audio-wave',
  templateUrl: './audio-wave.component.html',
  styleUrls: ['./audio-wave.component.scss'],
})
export class AudioWaveComponent {
  @Input() isActive = false;
  @Input() color = '#1a73e8';
  bars = [0, 1, 2, 3, 4, 5, 6];
}

