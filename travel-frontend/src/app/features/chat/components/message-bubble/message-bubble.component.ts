import { Component, Input, OnChanges } from '@angular/core';
import { Message, VoiceIntent } from '../../../../models/message.model';

export type ViewMode = 'optimised' | 'summary';

export interface ResponseCard {
  intent: VoiceIntent;
  icon:   string;
  title:  string;
  body:   string;
}

const INTENT_META: Record<string, { icon: string; title: string }> = {
  weather:     { icon: '🌤️', title: 'Weather'      },
  flight:      { icon: '✈️',  title: 'Flights'      },
  attractions: { icon: '🗺️', title: 'Attractions'  },
  currency:    { icon: '💱',  title: 'Currency'     },
  timezone:    { icon: '🕐',  title: 'Local Time'   },
  general:     { icon: '💡',  title: 'Travel Info'  },
};

@Component({
  standalone: false,
  selector: 'app-message-bubble',
  templateUrl: './message-bubble.component.html',
  styleUrls: ['./message-bubble.component.scss'],
})
export class MessageBubbleComponent implements OnChanges {
  @Input() message!: Message;

  cards: ResponseCard[]  = [];
  isMultiIntent          = false;

  /** Per-message view mode — toggled by the tab buttons. */
  viewMode: ViewMode = 'optimised';

  ngOnChanges(): void {
    if (this.message.role === 'assistant' && !this.message.isLoading) {
      this.buildCards();
    }
  }

  private buildCards(): void {
    const intents = this.message.intents ?? (this.message.intent ? [this.message.intent] : []);
    this.isMultiIntent = intents.length > 1;

    if (!this.isMultiIntent) {
      this.cards = [];
      return;
    }

    const responses = this.message.agentResponses ?? {};

    this.cards = intents
      .filter(intent => responses[intent] && responses[intent].trim())
      .map(intent => ({
        intent,
        icon:  INTENT_META[intent]?.icon  ?? '📡',
        title: INTENT_META[intent]?.title ?? intent,
        body:  responses[intent].trim(),
      }));

    // Fallback: if no agentResponses, render merged text as one card
    if (this.cards.length === 0 && this.message.text) {
      const primary = intents[0] ?? 'general';
      this.cards = [{
        intent: primary as VoiceIntent,
        icon:   INTENT_META[primary]?.icon  ?? '💡',
        title:  'Response',
        body:   this.message.text,
      }];
    }
  }

  setViewMode(mode: ViewMode): void {
    this.viewMode = mode;
  }

  getIntentMeta(intent: string): { icon: string; title: string } {
    return INTENT_META[intent] ?? { icon: '📡', title: intent };
  }
}

