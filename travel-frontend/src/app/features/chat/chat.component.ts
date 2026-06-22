import { Component, OnInit } from '@angular/core';
import { ChatStateService } from '../../core/services/chat-state.service';

@Component({
  standalone: false,
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss'],
})
export class ChatComponent implements OnInit {
  constructor(private chatState: ChatStateService) {}

  ngOnInit(): void {
    // Seed welcome message
    this.chatState.addMessage({
      id: this.chatState.generateId(),
      role: 'assistant',
      text: "Hello! I'm your travel planning assistant. You can type or use the mic on the right. Ask me about weather, flights, attractions, currency, or time zones!",
      timestamp: new Date(),
      intent: 'general',
    });
  }
}

