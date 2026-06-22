import { Component, Input, OnChanges } from '@angular/core';

@Component({
  standalone: false,
  selector: 'app-toast',
  templateUrl: './toast.component.html',
  styleUrls: ['./toast.component.scss'],
})
export class ToastComponent implements OnChanges {
  @Input() message = '';
  @Input() type: 'success' | 'error' | 'info' = 'info';
  @Input() visible = false;
  showing = false;

  ngOnChanges(): void {
    if (this.visible && this.message) {
      this.showing = true;
      setTimeout(() => (this.showing = false), 3500);
    }
  }
}

