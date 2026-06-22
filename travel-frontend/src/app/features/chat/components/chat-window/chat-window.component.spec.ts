import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Component, Input } from '@angular/core';
import { of, BehaviorSubject } from 'rxjs';

import { ChatWindowComponent } from './chat-window.component';
import { ChatStateService } from '../../../../core/services/chat-state.service';
import { TextApiService } from '../../../../api/text-api.service';
import { SessionService } from '../../../../core/services/session.service';
import { LoggerService } from '../../../../core/services/logger.service';
import { Message } from '../../../../models/message.model';
import { VoiceState } from '../../../../models/voice-state.model';

// ---- Stubs ----
@Component({ selector: 'app-message-bubble', template: '' })
class MessageBubbleStub { @Input() message!: Message; }

const idleVoice: VoiceState = {
  status: 'idle', silenceCountdown: 4,
  liveTranscript: '', errorMessage: '', isTtsSpeaking: false,
};

describe('ChatWindowComponent', () => {
  let component: ChatWindowComponent;
  let fixture: ComponentFixture<ChatWindowComponent>;
  let messagesSubject: BehaviorSubject<Message[]>;
  let voiceSubject: BehaviorSubject<VoiceState>;
  let chatStateSpy: jasmine.SpyObj<ChatStateService>;
  let textApiSpy: jasmine.SpyObj<TextApiService>;
  let sessionSpy: jasmine.SpyObj<SessionService>;
  let loggerSpy: jasmine.SpyObj<LoggerService>;

  beforeEach(async () => {
    messagesSubject = new BehaviorSubject<Message[]>([]);
    voiceSubject    = new BehaviorSubject<VoiceState>(idleVoice);

    chatStateSpy = jasmine.createSpyObj('ChatStateService', [
      'addMessage', 'updateMessage', 'generateId', 'clearActiveToolEvents', 'addToolEvent',
    ], {
      messages$:         messagesSubject.asObservable(),
      voiceState$:       voiceSubject.asObservable(),
      activeToolEvents$: of([]),
    });
    chatStateSpy.generateId.and.returnValue('gen-id');

    textApiSpy  = jasmine.createSpyObj('TextApiService', ['sendText']);
    sessionSpy  = jasmine.createSpyObj('SessionService', ['getSessionId']);
    loggerSpy   = jasmine.createSpyObj('LoggerService', ['log', 'error', 'warn']);
    sessionSpy.getSessionId.and.returnValue('sess-1');

    await TestBed.configureTestingModule({
      declarations: [ChatWindowComponent, MessageBubbleStub],
      imports: [CommonModule, FormsModule],
      providers: [
        { provide: ChatStateService, useValue: chatStateSpy },
        { provide: TextApiService,   useValue: textApiSpy   },
        { provide: SessionService,   useValue: sessionSpy   },
        { provide: LoggerService,    useValue: loggerSpy    },
      ],
    }).compileComponents();

    fixture   = TestBed.createComponent(ChatWindowComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should render the chat-window wrapper', () => {
    expect(fixture.nativeElement.querySelector('.chat-window')).toBeTruthy();
  });

  it('sendText should do nothing when inputText is empty', () => {
    component.inputText = '   ';
    component.sendText();
    expect(chatStateSpy.addMessage).not.toHaveBeenCalled();
  });

  it('sendText should add user + loading messages and call textApi', () => {
    textApiSpy.sendText.and.returnValue(of({ response: 'Hi', intent: 'general', tool_events: [] }));
    component.inputText = 'Hello';
    component.sendText();
    expect(chatStateSpy.addMessage).toHaveBeenCalledTimes(2);
    expect(textApiSpy.sendText).toHaveBeenCalledWith('Hello', 'sess-1');
    expect(component.inputText).toBe('');
  });

  it('sendText should update loading message with response on success', () => {
    textApiSpy.sendText.and.returnValue(of({
      response: 'Sunny', intent: 'weather', tool_events: [],
    }));
    component.inputText = 'Weather?';
    component.sendText();
    expect(chatStateSpy.updateMessage).toHaveBeenCalled();
    expect(component.isSubmitting).toBeFalse();
  });

  it('isVoiceActive returns true for listening status', () => {
    const vs: VoiceState = { ...idleVoice, status: 'listening' };
    expect(component.isVoiceActive(vs)).toBeTrue();
  });

  it('isVoiceActive returns false for idle status', () => {
    expect(component.isVoiceActive(idleVoice)).toBeFalse();
  });

  it('onKeydown Enter without Shift calls sendText', () => {
    spyOn(component, 'sendText');
    component.onKeydown(new KeyboardEvent('keydown', { key: 'Enter', shiftKey: false }));
    expect(component.sendText).toHaveBeenCalled();
  });

  it('onKeydown Shift+Enter does not call sendText', () => {
    spyOn(component, 'sendText');
    component.onKeydown(new KeyboardEvent('keydown', { key: 'Enter', shiftKey: true }));
    expect(component.sendText).not.toHaveBeenCalled();
  });

  it('stt-badge appears when status is listening', () => {
    voiceSubject.next({ ...idleVoice, status: 'listening' });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.stt-badge')).toBeTruthy();
  });

  it('stt-badge hidden when status is idle', () => {
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.stt-badge')).toBeNull();
  });

  it('trackById returns message id', () => {
    const msg: Message = { id: 'abc', role: 'user', text: 'hi', timestamp: new Date() };
    expect(component.trackById(0, msg)).toBe('abc');
  });
});
