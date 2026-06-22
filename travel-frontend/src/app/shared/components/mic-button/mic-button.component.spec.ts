import { ComponentFixture, TestBed } from '@angular/core/testing';
import { MicButtonComponent } from './mic-button.component';
import { CommonModule } from '@angular/common';

describe('MicButtonComponent', () => {
  let component: MicButtonComponent;
  let fixture: ComponentFixture<MicButtonComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [MicButtonComponent],
      imports: [CommonModule],
    }).compileComponents();
    fixture = TestBed.createComponent(MicButtonComponent);
    component = fixture.componentInstance;
  });

  it('should show Tap to Speak when idle', () => {
    component.status = 'idle';
    expect(component.label).toBe('Tap to Speak');
    expect(component.icon).toBe('🎙️');
  });

  it('should show Listening when active', () => {
    component.status = 'listening';
    expect(component.label).toBe('Listening...');
    expect(component.isActive).toBeTrue();
  });

  it('should emit micClick on button click', () => {
    spyOn(component.micClick, 'emit');
    component.status = 'idle';
    fixture.detectChanges();
    fixture.nativeElement.querySelector('.mic-btn').click();
    expect(component.micClick.emit).toHaveBeenCalled();
  });

  it('should return stop icon when silence_countdown', () => {
    component.status = 'silence_countdown';
    expect(component.label).toBe('Sending soon...');
    expect(component.icon).toBe('⏹');
    expect(component.isActive).toBeTrue();
  });

  it('should return hourglass icon when processing', () => {
    component.status = 'processing';
    expect(component.label).toBe('Processing...');
    expect(component.icon).toBe('⏳');
    expect(component.isActive).toBeFalse();
  });

  it('should return speaker icon when speaking', () => {
    component.status = 'speaking';
    expect(component.label).toBe('Speaking...');
    expect(component.icon).toBe('🔊');
    expect(component.isActive).toBeFalse();
  });

  it('should disable button when processing', () => {
    component.status = 'processing';
    fixture.detectChanges();
    const btn: HTMLButtonElement = fixture.nativeElement.querySelector('.mic-btn');
    expect(btn.disabled).toBeTrue();
  });

  it('should not disable button when idle', () => {
    component.status = 'idle';
    fixture.detectChanges();
    const btn: HTMLButtonElement = fixture.nativeElement.querySelector('.mic-btn');
    expect(btn.disabled).toBeFalse();
  });
});
