import { TestBed } from '@angular/core/testing';
import { ChatStateService } from './chat-state.service';

describe('ChatStateService', () => {
  let service: ChatStateService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ChatStateService);
  });

  it('should add message and emit via messages$', (done) => {
    const msg = {
      id: 'test1', role: 'user' as any,
      text: 'Hello', timestamp: new Date()
    };
    service.messages$.subscribe(msgs => {
      if (msgs.length > 0) {
        expect(msgs[0].text).toBe('Hello');
        done();
      }
    });
    service.addMessage(msg);
  });

  it('should update voice state partially', () => {
    service.setVoiceState({ status: 'listening' });
    service.voiceState$.subscribe(state => {
      expect(state.status).toBe('listening');
    });
  });

  it('should add and clear tool events', () => {
    service.addToolEvent({
      id: 't1', toolName: 'get_weather', label: 'Fetching',
      status: 'running', timestamp: new Date()
    });
    service.activeToolEvents$.subscribe(events => {
      if (events.length > 0) expect(events[0].toolName).toBe('get_weather');
    });
    service.clearActiveToolEvents();
    service.activeToolEvents$.subscribe(e => expect(e.length).toBe(0));
  });

  it('generateId() should return unique ids', () => {
    const ids = new Set([service.generateId(), service.generateId(), service.generateId()]);
    expect(ids.size).toBe(3);
  });

  it('getMessages() should return current snapshot', () => {
    const msg = { id: 'm1', role: 'user' as any, text: 'hi', timestamp: new Date() };
    service.addMessage(msg);
    expect(service.getMessages().length).toBe(1);
  });

  it('clearConversation() should reset messages and voice state', () => {
    service.addMessage({ id: 'm1', role: 'user' as any, text: 'hi', timestamp: new Date() });
    service.setVoiceState({ status: 'listening' });
    service.clearConversation();
    expect(service.getMessages().length).toBe(0);
    service.voiceState$.subscribe(s => expect(s.status).toBe('idle'));
  });
});
