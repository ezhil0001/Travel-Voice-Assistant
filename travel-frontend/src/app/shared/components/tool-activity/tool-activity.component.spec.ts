import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CommonModule } from '@angular/common';
import { ToolActivityComponent } from './tool-activity.component';
import { ToolEvent } from '../../../models/message.model';

const makeEvent = (overrides: Partial<ToolEvent> = {}): ToolEvent => ({
  id: 'e1',
  toolName: 'get_weather',
  label: 'Fetching weather data',
  status: 'success',
  timestamp: new Date('2026-01-01T12:00:00'),
  ...overrides,
});

describe('ToolActivityComponent', () => {
  let component: ToolActivityComponent;
  let fixture: ComponentFixture<ToolActivityComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ToolActivityComponent],
      imports: [CommonModule],
    }).compileComponents();
    fixture = TestBed.createComponent(ToolActivityComponent);
    component = fixture.componentInstance;
  });

  it('should not render when events is empty', () => {
    component.events = [];
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.tool-activity')).toBeNull();
  });

  it('should render a header row for each event', () => {
    component.events = [makeEvent({ id: 'e1' }), makeEvent({ id: 'e2', toolName: 'get_flights' })];
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelectorAll('.event-header').length).toBe(2);
  });

  it('toggle should expand then collapse an event', () => {
    expect(component.isExpanded('e1')).toBeFalse();
    component.toggle('e1');
    expect(component.isExpanded('e1')).toBeTrue();
    component.toggle('e1');
    expect(component.isExpanded('e1')).toBeFalse();
  });

  it('getStatusIcon returns correct icons', () => {
    expect(component.getStatusIcon('success')).toBe('✅');
    expect(component.getStatusIcon('error')).toBe('❌');
    expect(component.getStatusIcon('running')).toBe('⏳');
  });

  it('getToolIcon returns emoji for known tools and fallback for unknown', () => {
    expect(component.getToolIcon('get_weather')).toBe('🌤');
    expect(component.getToolIcon('get_flights')).toBe('✈️');
    expect(component.getToolIcon('unknown_tool')).toBe('📡');
  });

  it('should show detail panel after toggling', () => {
    component.events = [makeEvent({ id: 'e1', detail: 'city=Tokyo' })];
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.event-detail')).toBeNull();
    component.toggle('e1');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.event-detail')).toBeTruthy();
  });

  it('should not render error-detail row when errorMessage is absent', () => {
    component.events = [makeEvent({ id: 'e1' })];
    component.toggle('e1');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.error-detail')).toBeNull();
  });

  it('should render error-detail row when errorMessage is present', () => {
    component.events = [makeEvent({ id: 'e1', status: 'error', errorMessage: 'timeout' })];
    component.toggle('e1');
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelector('.error-detail')).toBeTruthy();
  });
});
