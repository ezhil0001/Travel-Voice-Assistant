import { Component, Input } from '@angular/core';
import { ToolEvent } from '../../../models/message.model';

@Component({
  standalone: false,
  selector: 'app-tool-activity',
  templateUrl: './tool-activity.component.html',
  styleUrls: ['./tool-activity.component.scss'],
})
export class ToolActivityComponent {
  @Input() events: ToolEvent[] = [];
  expandedIds = new Set<string>();

  toggle(id: string): void {
    if (this.expandedIds.has(id)) this.expandedIds.delete(id);
    else this.expandedIds.add(id);
  }

  isExpanded(id: string): boolean {
    return this.expandedIds.has(id);
  }

  getStatusIcon(status: string): string {
    if (status === 'success') return '✅';
    if (status === 'error')   return '❌';
    return '⏳';
  }

  getToolIcon(toolName: string): string {
    const icons: Record<string, string> = {
      get_weather:     '🌤',
      get_flights:     '✈️',
      get_attractions: '🗺',
      get_currency:    '💱',
      get_timezone:    '🕐',
    };
    return icons[toolName] || '📡';
  }
}
