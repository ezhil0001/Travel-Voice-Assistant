import { ComponentFixture, TestBed } from '@angular/core/testing';
import { CommonModule } from '@angular/common';
import { ToastComponent } from './toast.component';

describe('ToastComponent', () => {
  let component: ToastComponent;
  let fixture: ComponentFixture<ToastComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ToastComponent],
      imports: [CommonModule],
    }).compileComponents();
    fixture = TestBed.createComponent(ToastComponent);
    component = fixture.componentInstance;
  });

  it('should start with showing = false', () => {
    fixture.detectChanges();
    expect(component.showing).toBeFalse();
  });

  it('should set showing = true when visible and message are set', () => {
    component.message = 'Hello';
    component.visible = true;
    component.ngOnChanges();
    expect(component.showing).toBeTrue();
  });

  it('should not show when message is empty', () => {
    component.message = '';
    component.visible = true;
    component.ngOnChanges();
    expect(component.showing).toBeFalse();
  });

  it('should not show when visible is false', () => {
    component.message = 'Hello';
    component.visible = false;
    component.ngOnChanges();
    expect(component.showing).toBeFalse();
  });

  it('should apply type-error class for error type', () => {
    component.message = 'Oops';
    component.type = 'error';
    component.visible = true;
    component.ngOnChanges();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement.querySelector('.toast');
    expect(el.classList.contains('type-error')).toBeTrue();
  });

  it('should apply type-success class for success type', () => {
    component.message = 'Done';
    component.type = 'success';
    component.visible = true;
    component.ngOnChanges();
    fixture.detectChanges();
    const el: HTMLElement = fixture.nativeElement.querySelector('.toast');
    expect(el.classList.contains('type-success')).toBeTrue();
  });
});
