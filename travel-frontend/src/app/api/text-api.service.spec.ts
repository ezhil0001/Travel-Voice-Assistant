import { TestBed } from '@angular/core/testing';
import {
  HttpClientTestingModule,
  HttpTestingController,
} from '@angular/common/http/testing';

import { TextApiService } from './text-api.service';

describe('TextApiService', () => {
  let service: TextApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      imports: [HttpClientTestingModule],
      providers: [TextApiService],
    });
    service  = TestBed.inject(TextApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => httpMock.verify());

  it('should POST to /text/query with text and session_id', () => {
    service.sendText('weather in Tokyo', 'sess_001').subscribe();

    const req = httpMock.expectOne('http://localhost:8000/text/query');
    expect(req.request.method).toBe('POST');
    expect(req.request.body).toEqual({ text: 'weather in Tokyo', session_id: 'sess_001' });
    req.flush({ response: 'Tokyo is 18C', intent: 'weather', tool_events: [] });
  });

  it('should return response text and intent from the backend', () => {
    service.sendText('weather in Tokyo', 'sess_001').subscribe(res => {
      expect(res.response).toBe('Tokyo is 18C');
      expect(res.intent).toBe('weather');
    });

    httpMock
      .expectOne('http://localhost:8000/text/query')
      .flush({ response: 'Tokyo is 18C', intent: 'weather', tool_events: [] });
  });

  it('should always return an array for tool_events even when backend omits it', () => {
    service.sendText('hello', 'sess_002').subscribe(res => {
      expect(Array.isArray(res.tool_events)).toBeTrue();
      expect(res.tool_events.length).toBe(0);
    });

    // Simulate a backend that doesn't yet include tool_events in its payload
    httpMock
      .expectOne('http://localhost:8000/text/query')
      .flush({ response: 'Hi there!', intent: 'general' });
  });

  it('should map tool_events entries when they are present', () => {
    service.sendText('weather in Tokyo', 'sess_001').subscribe(res => {
      expect(res.tool_events.length).toBe(1);
      expect(res.tool_events[0].tool_name).toBe('get_weather');
      expect(res.tool_events[0].status).toBe('success');
    });

    httpMock
      .expectOne('http://localhost:8000/text/query')
      .flush({
        response: 'Tokyo is 18C',
        intent: 'weather',
        tool_events: [
          { tool_name: 'get_weather', label: 'Fetching weather', status: 'success' },
        ],
      });
  });

  it('should default intent to "general" when backend omits it', () => {
    service.sendText('hello', 'sess_003').subscribe(res => {
      expect(res.intent).toBe('general');
    });

    httpMock
      .expectOne('http://localhost:8000/text/query')
      .flush({ response: 'Hi!', tool_events: [] });
  });
});
