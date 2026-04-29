const WebSocket = require('ws');

const WS_URL = 'ws://localhost:8080';
const TIMEOUT_MS = 20000;

const ws = new WebSocket(WS_URL);
let sampleCount = 0;
let switched = false;
const preRatios = [];
const postRatios = [];
const startedAt = Date.now();

function avg(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

const timeout = setTimeout(() => {
  console.error('[check] Timeout waiting eeg_data');
  try { ws.close(); } catch {}
  process.exit(1);
}, TIMEOUT_MS);

ws.on('open', () => {
  console.log('[check] WS open');
  ws.send(JSON.stringify({ type: 'start_session', payload: {} }));
});

ws.on('message', (buffer) => {
  const msg = JSON.parse(buffer.toString());

  if (msg.type === 'server_hello') {
    console.log('[check] server_hello', msg.version);
    return;
  }

  if (msg.type === 'session_started') {
    console.log('[check] session_started', msg.sessionId);
    return;
  }

  if (msg.type === 'error') {
    console.error('[check] server_error', msg.message);
    clearTimeout(timeout);
    ws.close();
    process.exit(1);
  }

  if (msg.type !== 'eeg_data') return;

  sampleCount += 1;
  const ratio = Number(msg?.thetaBeta?.ratio ?? 0);

  if (!switched) {
    preRatios.push(ratio);
  } else {
    postRatios.push(ratio);
  }

  if (sampleCount === 6 && !switched) {
    switched = true;
    ws.send(JSON.stringify({ type: 'set_trance_mode', payload: { enabled: true } }));
    console.log('[check] trance_mode ON at sample', sampleCount);
  }

  if (sampleCount >= 16) {
    const result = {
      sampleCount,
      avgRatioPre: Number(avg(preRatios).toFixed(3)),
      avgRatioPost: Number(avg(postRatios).toFixed(3)),
      lastAction: msg?.command?.action,
      lastIntensity: Number(msg?.command?.intensity ?? 0),
      elapsedMs: Date.now() - startedAt,
    };

    console.log('[check] result', JSON.stringify(result, null, 2));
    ws.send(JSON.stringify({ type: 'stop_session' }));
    clearTimeout(timeout);
    ws.close();
  }
});

ws.on('close', () => {
  console.log('[check] WS closed');
  process.exit(0);
});

ws.on('error', (err) => {
  console.error('[check] WS error', err.message);
  clearTimeout(timeout);
  process.exit(1);
});
