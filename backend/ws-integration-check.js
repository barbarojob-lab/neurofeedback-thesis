const WebSocket = require('ws');
const path = require('path');

const WS_URL = 'ws://localhost:8080';
const TIMEOUT_MS = 90000;
const TARGET_SAMPLES = 10;
const datasetArg = process.argv[2];
const datasetPath = datasetArg
  ? path.resolve(datasetArg)
  : path.resolve(__dirname, '..', 'data', 'subject_11_high_test.edf');

const ws = new WebSocket(WS_URL);
let sampleCount = 0;
let datasetLoaded = false;
let sessionStarted = false;
const ratios = [];
const startedAt = Date.now();

function avg(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

const timeout = setTimeout(() => {
  console.error('[check] Timeout waiting integration flow');
  try { ws.close(); } catch {}
  process.exit(1);
}, TIMEOUT_MS);

ws.on('open', () => {
  console.log('[check] WS open');
});

ws.on('message', (buffer) => {
  const msg = JSON.parse(buffer.toString());

  if (msg.type === 'server_hello') {
    console.log('[check] server_hello', msg.version);
    ws.send(JSON.stringify({ type: 'load_dataset', payload: { path: datasetPath } }));
    return;
  }

  if (msg.type === 'dataset_loaded') {
    datasetLoaded = true;
    console.log('[check] dataset_loaded', msg.dataset?.sampleRate, 'Hz');
    ws.send(JSON.stringify({ type: 'start_session', payload: {} }));
    return;
  }

  if (msg.type === 'session_started') {
    sessionStarted = true;
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
  ratios.push(ratio);

  if (sampleCount >= TARGET_SAMPLES) {
    const result = {
      datasetLoaded,
      sessionStarted,
      sampleCount,
      avgRatio: Number(avg(ratios).toFixed(3)),
      predictedLabel: msg?.state_prediction?.predicted_label ?? null,
      confidence: Number(msg?.state_prediction?.confidence ?? 0),
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
