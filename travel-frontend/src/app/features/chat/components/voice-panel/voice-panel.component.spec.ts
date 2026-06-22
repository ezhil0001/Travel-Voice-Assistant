import { ComponentFixture, TestBed, fakeAsync, tick } from '@angular/core/testing';
import { CommonModule } from '@angular/common';
import { Component, Input, Output, EventEmitter } from '@angular/core';
import { of } from 'rxjs';

import { VoicePanelComponent } from './voice-panel.component';
import { VoiceApiService } from '../../../../api/voice-api.service';
import { ChatStateService } from '../../../../core/services/chat-state.service';
import { SessionService } from '../../../../core/services/session.service';
import { LoggerService } from '../../../../core/services/logger.service';
import { ToolEvent } from '../../../../models/message.model';
import { VoiceState } from '../../../../models/voice-state.model';
import { BehaviorSubject } from 'rxjs';

// ---- Stubs ----
@Component({ selector: 'app-mic-button', template: '' })
class MicButtonStub {
  @Input() status = 'idle';
  @Output() micClick = new EventEmitter<void>();
}

@Component({ selector: 'app-audio-wave', template: '' })
class AudioWaveStub {
  @Input() isActive = false;
  @Input() color = '';
}

@Component({ selector: 'app-tool-activity', template: '' })
class ToolActivityStub {
  @Input() events: ToolEvent[] = [];
}

const idleState: VoiceState = {
  status: 'idle', silenceCountdown: 4,
  liveTranscript: '', errorMessage: '', isTtsSpeaking: false,
};

function makeState(patch: Partial<VoiceState> = {}): VoiceState {
  return { ...idleState, ...patch };
}

describe('VoicePanelComponent', () => {
  let component: VoicePanelComponent;
  let fixture: ComponentFixture<VoicePanelComponent>;
  let voiceStateSub: BehaviorSubject<VoiceState>;
  let toolEventsSub: BehaviorSubject<ToolEvent[]>;
  let chatStateSpy: jasmine.SpyObj<ChatStateService>;
  let sessionSpy: jasmine.SpyObj<SessionService>;
  let loggerSpy: jasmine.SpyObj<LoggerService>;
  let voiceApiSpy: jasmine.SpyObj<VoiceApiService>;

  beforeEach(async () => {
    voiceStateSub = new BehaviorSubject<VoiceState>(idleState);
    toolEventsSub = new BehaviorSubject<ToolEvent[]>([]);

    chatStateSpy = jasmine.createSpyObj('ChatStateService', [
      'setVoiceState', 'clearActiveToolEvents', 'addMessage', 'updateMessage',
      'addToolEvent', 'generateId', 'clearConversation',
    ], {
      voiceState$: voiceStateSub.asObservable(),
      activeToolEvents$: toolEventsSub.asObservable(),
    });
    chatStateSpy.generateId.and.returnValue('test-id');

    sessionSpy = jasmine.createSpyObj('SessionService', ['getSessionId']);
    sessionSpy.getSessionId.and.returnValue('sess-1');

    loggerSpy = jasmine.createSpyObj('LoggerService', ['log', 'error', 'warn']);

    voiceApiSpy = jasmine.createSpyObj('VoiceApiService', ['sendAudio']);

    await TestBed.configureTestingModule({
      declarations: [
        VoicePanelComponent,
        MicButtonStub,
        AudioWaveStub,
        ToolActivityStub,
      ],
      imports: [CommonModule],
      providers: [
        { provide: VoiceApiService, useValue: voiceApiSpy },
        { provide: ChatStateService, useValue: chatStateSpy },
        { provide: SessionService, useValue: sessionSpy },
        { provide: LoggerService, useValue: loggerSpy },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(VoicePanelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should render the panel title', () => {
    const title = fixture.nativeElement.querySelector('.panel-title');
    expect(title.textContent).toContain('Voice');
  });

  it('should show countdown bar only in silence_countdown status', () => {
    voiceStateSub.next(makeState({ status: 'silence_countdown', silenceCountdown: 2 }));
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.countdown-bar')).toBeTruthy();

    voiceStateSub.next(idleState);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.countdown-bar')).toBeNull();
  });

  it('should show skip button only when isTtsSpeaking', () => {
    voiceStateSub.next(makeState({ status: 'speaking', isTtsSpeaking: true }));
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.skip-btn')).toBeTruthy();

    voiceStateSub.next(idleState);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.skip-btn')).toBeNull();
  });

  it('should show error message only in error status', () => {
    voiceStateSub.next(makeState({ status: 'error', errorMessage: 'Mic denied' }));
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.error-msg').textContent).toContain('Mic denied');

    voiceStateSub.next(idleState);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.error-msg')).toBeNull();
  });

  it('skipTts should set voice state to idle', () => {
    component.skipTts();
    expect(chatStateSpy.setVoiceState).toHaveBeenCalledWith({ status: 'idle', isTtsSpeaking: false });
    expect(loggerSpy.log).toHaveBeenCalledWith('VoicePanel', 'TTS skipped by user');
  });

  it('should show no-tools placeholder when events is empty', () => {
    toolEventsSub.next([]);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.no-tools')).toBeTruthy();
  });

  it('should hide no-tools placeholder when events exist', () => {
    const event: ToolEvent = {
      id: 'e1', toolName: 'get_weather', label: 'Fetching weather',
      status: 'success', timestamp: new Date(),
    };
    toolEventsSub.next([event]);
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.no-tools')).toBeNull();
  });

  it('ngOnDestroy should clear intervals without throwing', () => {
    expect(() => component.ngOnDestroy()).not.toThrow();
  });
});
