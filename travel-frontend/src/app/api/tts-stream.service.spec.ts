import { TestBed } from '@angular/core/testing';
import { TtsStreamService, TtsEvent } from './tts-stream.service';

function makeFakeWs(overrides: Partial<WebSocket> = {}): WebSocket {
  return {
    readyState:     WebSocket.OPEN,
    bufferedAmount: 0,
    send:           jasmine.createSpy('send'),
    close:          jasmine.createSpy('close'),
    onmessage:      null as any,
    onerror:        null as any,
    onclose:        null as any,
    ...overrides,
  } as unknown as WebSocket;
}

describe('TtsStreamService', () => {
  let service: TtsStreamService;
  let wsSpy: jasmine.Spy;
  let fakeWs: WebSocket;

  beforeEach(() => {
    fakeWs = makeFakeWs();
    wsSpy = spyOn(window, 'WebSocket').and.returnValue(fakeWs);

    TestBed.configureTestingModule({});
    service = TestBed.inject(TtsStreamService);
  });

  afterEach(() => service.ngOnDestroy());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('connect() opens a WebSocket to /tts/stream', () => {
    service.connect();
    expect(wsSpy).toHaveBeenCalledWith(jasmine.stringContaining('/tts/stream'));
  });

  it('speak() sends a start_tts JSON frame with text and defaults', () => {
    service.connect();
    service.speak('Tokyo is 18°C today.').subscribe();
    const sent = JSON.parse((fakeWs.send as jasmine.Spy).calls.mostRecent().args[0]);
    expect(sent.type).toBe('start_tts');
    expect(sent.text).toBe('Tokyo is 18°C today.');
    expect(sent.language_code).toBe('en-IN');
    expect(sent.speaker).toBe('meera');
  });

  it('speak() sends custom language_code and speaker when provided', () => {
    service.connect();
    service.speak('नमस्ते', { languageCode: 'hi-IN', speaker: 'anushka' }).subscribe();
    const sent = JSON.parse((fakeWs.send as jasmine.Spy).calls.mostRecent().args[0]);
    expect(sent.language_code).toBe('hi-IN');
    expect(sent.speaker).toBe('anushka');
  });

  it('stop() sends a stop_tts frame', () => {
    service.connect();
    service.stop();
    const sent = JSON.parse((fakeWs.send as jasmine.Spy).calls.mostRecent().args[0]);
    expect(sent.type).toBe('stop_tts');
  });

  it('events observable emits tts_ready event', (done) => {
    service.connect();
    service.events.subscribe((evt: TtsEvent) => {
      if (evt.type === 'tts_ready') done();
    });
    (fakeWs.onmessage as any)({ data: JSON.stringify({ type: 'tts_ready' }) });
  });

  it('events observable emits tts_audio with audio_base64', (done) => {
    service.connect();
    service.events.subscribe((evt: TtsEvent) => {
      if (evt.type === 'tts_audio') {
        expect(evt.audio_base64).toBe('AAEC');
        expect(evt.format).toBe('wav');
        done();
      }
    });
    (fakeWs.onmessage as any)({
      data: JSON.stringify({ type: 'tts_audio', audio_base64: 'AAEC', format: 'wav' }),
    });
  });

  it('speak() Observable completes on tts_done', (done) => {
    service.connect();
    service.speak('hello').subscribe({ complete: done });
    // Simulate tts_ready then tts_done
    (fakeWs.onmessage as any)({ data: JSON.stringify({ type: 'tts_ready' }) });
    (fakeWs.onmessage as any)({ data: JSON.stringify({ type: 'tts_done' }) });
  });

  it('speak() Observable completes on tts_stopped', (done) => {
    service.connect();
    service.speak('hello').subscribe({ complete: done });
    (fakeWs.onmessage as any)({ data: JSON.stringify({ type: 'tts_stopped' }) });
  });

  it('decodeAudioChunk() decodes base64 to ArrayBuffer', () => {
    service.connect();
    // base64("ABC") = "QUJD"
    const buf = service.decodeAudioChunk('QUJD');
    expect(buf.byteLength).toBe(3);
    expect(new Uint8Array(buf)[0]).toBe(65); // 'A'
  });

  it('disconnect() closes the WebSocket', () => {
    service.connect();
    service.disconnect();
    expect(fakeWs.close).toHaveBeenCalledWith(1000, 'client-disconnect');
    expect(service.isConnected).toBeFalse();
  });
});
