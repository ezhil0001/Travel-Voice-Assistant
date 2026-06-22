import { Component, Input, OnChanges } from '@angular/core';
import { ToolEvent } from '../../../models/message.model';

@Component({
  standalone: false,
  selector: 'app-tool-activity',
  templateUrl: './tool-activity.component.html',
  styleUrls: ['./tool-activity.component.scss'],
})
export class ToolActivityComponent implements OnChanges {
  @Input() events: ToolEvent[] = [];

  panelOpen  = false;
  expandedId: string | null = null;

  get hasError(): boolean {
    return this.events.some(e => e.status === 'error');
  }

  get totalDuration(): number {
    return this.events.reduce((s, e) => s + (e.durationMs || 0), 0);
  }

  ngOnChanges(): void {
    this.expandedId = null;
  }

  togglePanel(): void {
    this.panelOpen = !this.panelOpen;
  }

  toggleEvent(id: string): void {
    this.expandedId = this.expandedId === id ? null : id;
  }

  isExpanded(id: string): boolean {
    return this.expandedId === id;
  }

  getToolIcon(toolName: string): string {
    const icons: Record<string, string> = {
      get_weather:     '🌤️',
      get_flights:     '✈️',
      get_attractions: '🗺️',
      get_currency:    '💱',
      get_timezone:    '🕐',
      general_llm:     '🧠',
    };
    return icons[toolName] || '📡';
  }

  formatDuration(ms?: number): string {
    if (!ms) return '—';
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  getStatusLabel(status: string): string {
    if (status === 'success') return 'Success';
    if (status === 'error')   return 'Failed';
    return 'Running';
  }
}
