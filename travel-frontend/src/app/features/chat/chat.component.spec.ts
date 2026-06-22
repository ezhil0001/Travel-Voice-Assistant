import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ChatComponent } from './chat.component';
import { ChatStateService } from '../../core/services/chat-state.service';
import { Component } from '@angular/core';

// Stubs for child components
@Component({ selector: 'app-chat-window', template: '' })
class ChatWindowStub {}

@Component({ selector: 'app-voice-panel', template: '' })
class VoicePanelStub {}

describe('ChatComponent', () => {
  let component: ChatComponent;
  let fixture: ComponentFixture<ChatComponent>;
  let chatStateSpy: jasmine.SpyObj<ChatStateService>;

  beforeEach(async () => {
    chatStateSpy = jasmine.createSpyObj('ChatStateService', ['addMessage', 'generateId']);
    chatStateSpy.generateId.and.returnValue('welcome-id');

    await TestBed.configureTestingModule({
      declarations: [ChatComponent, ChatWindowStub, VoicePanelStub],
      providers: [{ provide: ChatStateService, useValue: chatStateSpy }],
    }).compileComponents();

    fixture   = TestBed.createComponent(ChatComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should seed a welcome message on init', () => {
    expect(chatStateSpy.addMessage).toHaveBeenCalledOnceWith(
      jasmine.objectContaining({
        id:    'welcome-id',
        role:  'assistant',
        intent: 'general',
      })
    );
  });

  it('welcome message text should mention travel assistant', () => {
    const call = chatStateSpy.addMessage.calls.first();
    expect(call.args[0].text).toContain('travel planning assistant');
  });

  it('should render left and right panels', () => {
    expect(fixture.nativeElement.querySelector('.left-panel')).toBeTruthy();
    expect(fixture.nativeElement.querySelector('.right-panel')).toBeTruthy();
  });

  it('should render the logo text', () => {
    const logo = fixture.nativeElement.querySelector('.logo');
    expect(logo.textContent).toContain('Travel Assistant');
  });
});
