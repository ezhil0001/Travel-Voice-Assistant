import { TestBed } from '@angular/core/testing';
import { SttStreamService, SttEvent } from './stt-stream.service';

// Helper: build a minimal fake WebSocket
function makeFakeWs(overrides: Partial<WebSocket> = {}): WebSocket {
  const ws = {
    readyState:     WebSocket.OPEN,
    bufferedAmount: 0,
    binaryType:     'arraybuffer',
    send:           jasmine.createSpy('send'),
    close:          jasmine.createSpy('close'),
    onmessage:      null as any,
    onerror:        null as any,
    onclose:        null as any,
    ...overrides,
  } as unknown as WebSocket;
  return ws;
}

describe('SttStreamService', () => {
  let service: SttStreamService;
  let wsSpy: jasmine.Spy;
  let fakeWs: WebSocket;

  beforeEach(() => {
    fakeWs = makeFakeWs();
    wsSpy = spyOn(window, 'WebSocket').and.returnValue(fakeWs);

    TestBed.configureTestingModule({});
    service = TestBed.inject(SttStreamService);
  });

  afterEach(() => service.ngOnDestroy());

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('connect() opens a WebSocket to /voice/stream', () => {
    service.connect();
    expect(wsSpy).toHaveBeenCalledWith(jasmine.stringContaining('/voice/stream'));
  });

  it('startSession() sends a start_stt JSON frame', () => {
    service.connect();
    service.startSession('sess-001', 'hi-IN');
    const sent = JSON.parse((fakeWs.send as jasmine.Spy).calls.mostRecent().args[0]);
    expect(sent.type).toBe('start_stt');
    expect(sent.session_id).toBe('sess-001');
    expect(sent.language_code).toBe('hi-IN');
  });

  it('stopSession() sends a stop_stt JSON frame', () => {
    service.connect();
    service.stopSession();
    const sent = JSON.parse((fakeWs.send as jasmine.Spy).calls.mostRecent().args[0]);
    expect(sent.type).toBe('stop_stt');
  });

  it('sendAudioChunk() sends binary data when socket is open', () => {
    service.connect();
    const buf = new ArrayBuffer(256);
    service.sendAudioChunk(buf);
    expect(fakeWs.send).toHaveBeenCalledWith(buf);
  });

  it('events observable emits parsed stt_ready event', (done) => {
    service.connect();
    service.events.subscribe((evt: SttEvent) => {
      if (evt.type === 'stt_ready') {
        done();
      }
    });
    // Simulate server message
    (fakeWs.onmessage as any)({ data: JSON.stringify({ type: 'stt_ready' }) });
  });

  it('events observable emits stt_interim with transcript', (done) => {
    service.connect();
    service.events.subscribe((evt: SttEvent) => {
      if (evt.type === 'stt_interim') {
        expect(evt.transcript).toBe('hello world');
        done();
      }
    });
    (fakeWs.onmessage as any)({
      data: JSON.stringify({ type: 'stt_interim', transcript: 'hello world', is_final: false }),
    });
  });

  it('isConnected returns true when socket is OPEN', () => {
    service.connect();
    expect(service.isConnected).toBeTrue();
  });

  it('disconnect() closes the socket', () => {
    service.connect();
    service.disconnect();
    expect(fakeWs.close).toHaveBeenCalledWith(1000, 'client-disconnect');
    expect(service.isConnected).toBeFalse();
  });
});
