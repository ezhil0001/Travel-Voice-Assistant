import { Component, Input } from '@angular/core';
import { Message } from '../../../../models/message.model';

@Component({
  standalone: false,
  selector: 'app-message-bubble',
  templateUrl: './message-bubble.component.html',
  styleUrls: ['./message-bubble.component.scss'],
})
export class MessageBubbleComponent {
  @Input() message!: Message;
}

