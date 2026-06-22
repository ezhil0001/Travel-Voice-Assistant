import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CommonModule } from '@angular/common';
import { Component, Input } from '@angular/core';
import { MessageBubbleComponent } from './message-bubble.component';
import { Message, ToolEvent } from '../../../../models/message.model';

// Stub ToolActivityComponent to isolate MessageBubble tests
@Component({ selector: 'app-tool-activity', template: '' })
class ToolActivityStubComponent {
  @Input() events: ToolEvent[] = [];
}

const makeMsg = (overrides: Partial<Message> = {}): Message => ({
  id: 'm1',
  role: 'user',
  text: 'Hello',
  timestamp: new Date('2026-01-01T10:30:00'),
  ...overrides,
});

describe('MessageBubbleComponent', () => {
  let component: MessageBubbleComponent;
  let fixture: ComponentFixture<MessageBubbleComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [MessageBubbleComponent, ToolActivityStubComponent],
      imports: [CommonModule],
    }).compileComponents();
    fixture = TestBed.createComponent(MessageBubbleComponent);
    component = fixture.componentInstance;
  });

  it('should apply .user class for user message', () => {
    component.message = makeMsg({ role: 'user' });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.message-row.user')).toBeTruthy();
  });

  it('should apply .assistant class for assistant message', () => {
    component.message = makeMsg({ role: 'assistant', text: 'Hi' });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.message-row.assistant')).toBeTruthy();
  });

  it('should show loading dots when isLoading is true', () => {
    component.message = makeMsg({ role: 'assistant', text: '', isLoading: true });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.dots')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('p')).toBeNull();
  });

  it('should show text and hide dots when not loading', () => {
    component.message = makeMsg({ role: 'assistant', text: 'Reply text' });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.dots')).toBeNull();
    expect(fixture.nativeElement.querySelector('p').textContent.trim()).toBe('Reply text');
  });

  it('should show intent badge on assistant message with intent', () => {
    component.message = makeMsg({ role: 'assistant', text: 'Sunny', intent: 'weather' });
    fixture.detectChanges();
    const badge = fixture.nativeElement.querySelector('.badge');
    expect(badge).toBeTruthy();
    expect(badge.textContent.trim()).toBe('weather');
  });

  it('should not show intent badge on user message', () => {
    component.message = makeMsg({ role: 'user', text: 'What is the weather?' });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.badge')).toBeNull();
  });

  it('should not show tool-activity when toolEvents is empty', () => {
    component.message = makeMsg({ role: 'assistant', text: 'Ok', toolEvents: [] });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('app-tool-activity')).toBeNull();
  });

  it('should show tool-activity when toolEvents are present', () => {
    const event: ToolEvent = {
      id: 'e1', toolName: 'get_weather', label: 'Fetching weather',
      status: 'success', timestamp: new Date(),
    };
    component.message = makeMsg({ role: 'assistant', text: 'Sunny', toolEvents: [event] });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('app-tool-activity')).toBeTruthy();
  });

  it('should hide footer when isLoading', () => {
    component.message = makeMsg({ role: 'assistant', text: '', isLoading: true });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.bubble-footer')).toBeNull();
  });
});
