/**
 * server.ts  —  backend/src/
 *
 * Orquestador principal del sistema de neurofeedback EEG.
 *
 * Responsabilidades:
 *   1. Servidor HTTP (health-check) + WebSocket server en puerto 8080.
 *   2. Gestión del ciclo de vida de sesión (start / stop / reset).
 *   3. Ejecución del pipeline DSP por cada sample del simulador EEG.
 *   4. Distribución del FeedbackPayload a todos los clientes WS conectados.
 *   5. Medición continua de latencia del pipeline (objetivo: < 5 ms).
 *
 * ── Arquitectura del pipeline (por sample) ───────────────────────────────
 *
 *   EEGSimulator (250 sps)
 *       │
 *       ▼  sample crudo [µV]
 *   NotchFilter            → elimina 50 Hz (interferencia red eléctrica)
 *       │
 *       ▼
 *   ButterworthBandpass    → pasa 1–40 Hz (banda EEG clínica)
 *       │                    [nota: highCut=40 en server, 30 en análisis strict —
 *       │                     40 Hz aquí para no cortar gamma en visualización;
 *       │                     BandPowerExtractor ignora gamma en feedback real-time]
 *       ▼
 *   SlidingWindow.push()
 *       │
 *       ├── (cada hopSize muestras, buffer lleno) ──────────────────────────────┐
 *       │                                                                       │
 *       ▼                                                                       ▼
 *   (acumular)                                                     applyHannWindow()
 *                                                                       │
 *                                                                       ▼
 *                                                               FFTAnalyzer.analyze()
 *                                                                       │
 *                                                                       ▼
 *                                                          BandPowerExtractor.extract()
 *                                                                       │
 *                                                          .computeThetaBetaRatio()
 *                                                                       │
 *                                                                       ▼
 *                                                          RunningZScore.push(tbr.ratio)
 *                                                                       │
 *                                                                       ▼
 *                                                          FeedbackEngine.computeCommand()
 *                                                                       │
 *                                                                       ▼
 *                                                          broadcast(FeedbackPayload) → WS clients
 *
 * ── Frecuencia de envío al frontend ─────────────────────────────────────
 *
 *   Con windowSize=256, hopSize=64, fs=250:
 *     Cada hop = 64/250 = 256 ms ≈ 4 epochs/segundo → ~250 ms entre payloads.
 *   El FeedbackPayload también incluye el filteredSample más reciente para
 *   que el frontend pueda dibujar la forma de onda EEG en tiempo real.
 *
 * ── Protocolo WebSocket ──────────────────────────────────────────────────
 *
 *   ENTRANTE (frontend → server):
 *     { type: 'start_session',    payload: SessionConfig      }
 *     { type: 'stop_session'                                  }
 *     { type: 'set_trance_mode',  payload: { enabled: boolean }}
 *     { type: 'set_inspection_channel', payload: { channel: EEGChannel }}
 *     { type: 'ping'                                          }
 *     { type: 'submit_subjective', payload: SubjectiveMeasure }
 *
 *   SALIENTE (server → frontend):
 *     { type: 'eeg_data',          ...FeedbackPayload         }
 *     { type: 'pong',              timestamp: number          }
 *     { type: 'session_started',   sessionId: string          }
 *     { type: 'session_stopped'                               }
 *     { type: 'inspection_channel_set', channel: EEGChannel   }
 *     { type: 'subjective_saved',  sessionId: string          }
 *     { type: 'error',             message: string            }
 */

import http     from "http";
import { WebSocketServer, WebSocket } from "ws";
import { parseDatasetMetadata, type DatasetMetadata } from "./datasets/parser";
import { DatasetReplayer } from "./datasets/replayer";

// ── Módulos DSP ──────────────────────────────────────────────────────────────
import { NotchFilter }         from "./filters/notch-filter";
import { ButterworthBandpass } from "./filters/butterworth-filter";
import { SlidingWindow }       from "./dsp/sliding-window";
import { FFTAnalyzer }         from "./dsp/fft-analyzer";
import { BandPowerExtractor }  from "./dsp/band-power";

// ── Módulos ML ──────────────────────────────────────────────────────────────
import { MLServiceClient, type MLServiceResult, type ClassifierPrediction } from "./ml-client";

// ── Tipos compartidos ────────────────────────────────────────────────────────
import type { BandPowers, ThetaBetaResult }  from "./dsp/band-power";
import type { SessionConfig } from "./types";

// ---------------------------------------------------------------------------
// Constantes de configuración
// ---------------------------------------------------------------------------

const PORT        = Number(process.env.PORT ?? 8080);
const SAMPLE_RATE = 250;   // Hz — OpenBCI Cyton default
const WINDOW_SIZE = 512;   // samples — 2.048 s de epoch (≈ 2 s, potencia de 2, igual que entrenamiento ML)
const HOP_SIZE    = 64;    // samples — 87.5 % de overlap → ~4 payloads/s

const EEG_CHANNELS = [
  "Fz", "Fp1", "Fp2", "F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2",
] as const;
const ML_CHANNELS = [
  "Fz", "Fp1", "F3", "C3", "Pz", "O1", "F4", "C4", "P4", "O2", "Cz",
] as const satisfies readonly typeof EEG_CHANNELS[number][];
type EEGChannel = typeof EEG_CHANNELS[number];
type ChannelSample = Record<EEGChannel, number>;

/**
 * Modo de adquisición de EEG:
 *   "simulator" — EEGSimulator (desarrollo y testing)
 *   "hardware"  — Esperar conexión de dispositivo real (OpenBCI, Muse, etc.)
 * Default: "hardware" para evitar simulación accidental en validación real
 */
const EEG_MODE = (process.env.EEG_MODE ?? "hardware") as "simulator" | "hardware";

/** Intervalo del log de latencia (ms) */
const LATENCY_LOG_INTERVAL_MS = 5_000;

/** Mínimo de samples del z-score antes de enviar comandos de feedback */
const ZSCORE_WARMUP_SAMPLES = 30;
const DEFAULT_FRONTAL_SPECIFICITY_THRESHOLD = 1.5;
const ARTIFACT_THRESHOLD_UV = 100;

// ---------------------------------------------------------------------------
// Tipos del protocolo WS
// ---------------------------------------------------------------------------

interface SubjectiveMeasure {
  sessionId : string;
  timestamp : number;
  label     : string;   // e.g. "relax_self_report"
  value     : number;   // 0–10 escala Likert
  notes?    : string;
}

interface FeedbackPayload {
  type            : "eeg_data";
  timestamp       : number;
  filteredSample  : number;        // último sample filtrado [µV] (compatibilidad)
  filteredSamples : number[];      // todos los samples del hop [µV] — para el osciloscopio
  frontalSpecificity      : number;
  frontalSpecificityValid : boolean;
  artifactDetected        : boolean;
  topographyTheta         : Record<EEGChannel, number>; // theta normalizada [0,1]
  inspection: {
    channel: EEGChannel;
    filteredSamples: number[];
    fftMagnitudes: number[];
    bandPowers: BandPowers;
  };
  bandPowers      : BandPowers;
  thetaBeta       : ThetaBetaResult;
  // ✅ NUEVO: Predicción del estado (reemplaza command)
  state_prediction: ClassifierPrediction | null;
  connectivity    : MLServiceResult | null;  // Resultados de conectividad del ML service
  pipelineMs      : number;        // tiempo de procesamiento del epoch [ms]
}

type IncomingMessage =
  | { type: "start_session";    payload: SessionConfig                }
  | { type: "stop_session"                                            }
  | { type: "set_trance_mode";  payload: { enabled: boolean }        }
  | { type: "set_inspection_channel"; payload: { channel: EEGChannel } }
  | { type: "load_dataset"; payload: { path: string }                 }
  | { type: "ping"                                                    }
  | { type: "submit_subjective"; payload: SubjectiveMeasure          };

// ---------------------------------------------------------------------------
// Estado de sesión
// ---------------------------------------------------------------------------

interface SessionState {
  id         : string;
  startedAt  : number;
  tranceMode : boolean;
  config     : SessionConfig;
}

// ---------------------------------------------------------------------------
// Simulador EEG (stub — reemplazar con el driver real del hardware)
// ---------------------------------------------------------------------------
// En producción este módulo se importaría desde ./hardware/eeg-simulator.
// Aquí se implementa como un generador de señal sintética para que el
// servidor sea autocontenido y testeable sin hardware.

class EEGSimulator {
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private tranceMode = false;
  private t = 0; // tiempo en muestras
  private blinkCountdown = 0;

  start(onSample: (sample: ChannelSample) => void): void {
    if (this.intervalId) return;
    const intervalMs = 1000 / SAMPLE_RATE; // 4 ms entre samples a 250 Hz

    this.intervalId = setInterval(() => {
      const sample = this._generateSampleFrame();
      onSample(sample);
      this.t++;
    }, intervalMs);
  }

  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  setTranceMode(enabled: boolean): void {
    this.tranceMode = enabled;
  }

  reset(): void {
    this.t = 0;
    this.blinkCountdown = 0;
  }

  private _generateSampleFrame(): ChannelSample {
    const frame = {} as ChannelSample;
    for (const channel of EEG_CHANNELS) {
      frame[channel] = this._generateChannelSample(channel);
    }
    return frame;
  }

  private _generateChannelSample(channel: EEGChannel): number {
    const fs = SAMPLE_RATE;
    const t  = this.t / fs;

    const thetaBase = this.tranceMode ? 15 : 3;
    const betaBase = this.tranceMode ? 3 : 5;
    const alphaBase = this.tranceMode ? 5 : 4;

    const thetaGainByChannel: Record<EEGChannel, number> = {
      Fz: 1.0,
      Fp1: 0.55,
      Fp2: 0.55,
      F3: 0.62,
      F4: 0.62,
      C3: 0.48,
      Cz: 0.5,
      C4: 0.48,
      P3: 0.35,
      Pz: 0.32,
      P4: 0.35,
      O1: 0.25,
      O2: 0.25,
    };
    const betaGainByChannel: Record<EEGChannel, number> = {
      Fz: 1.0,
      Fp1: 0.85,
      Fp2: 0.85,
      F3: 0.9,
      F4: 0.9,
      C3: 0.8,
      Cz: 0.82,
      C4: 0.8,
      P3: 0.7,
      Pz: 0.68,
      P4: 0.7,
      O1: 0.58,
      O2: 0.58,
    };

    const theta = thetaBase * thetaGainByChannel[channel] * Math.sin(2 * Math.PI * 6 * t);
    const beta = betaBase * betaGainByChannel[channel] * Math.sin(2 * Math.PI * 20 * t);
    const alpha = alphaBase * 0.7 * Math.sin(2 * Math.PI * 10 * t);
    const line = 2 * Math.sin(2 * Math.PI * 50 * t);
    let sample = theta + beta + alpha + line + this._noise(3.5);

    if (channel === "Fp1" || channel === "Fp2") {
      if (this.blinkCountdown > 0) {
        sample += 140 * Math.sin(2 * Math.PI * 1.8 * t);
        this.blinkCountdown -= 1;
      } else if (Math.random() < 0.003) {
        this.blinkCountdown = 10;
      }
    }

    return sample;
  }

  /** Ruido blanco gaussiano aproximado (Box-Muller) */
  private _noise(amplitude: number): number {
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    return amplitude * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

// ---------------------------------------------------------------------------
// Stub de SessionDB
// ---------------------------------------------------------------------------
// En producción importaría desde ./db/session-db.
// Stub para que el servidor compile y funcione sin base de datos.

class SessionDB {
  async insertSubjectiveMeasure(measure: SubjectiveMeasure): Promise<void> {
    console.log("[SessionDB] insertSubjectiveMeasure:", measure);
    // TODO: implementar persistencia real (SQLite / PostgreSQL)
  }
}

// ---------------------------------------------------------------------------
// Instancias del pipeline DSP (singleton por proceso)
// ---------------------------------------------------------------------------

interface ChannelPipelineState {
  notch: NotchFilter;
  bandpass: ButterworthBandpass;
  window: SlidingWindow;
  hopBuffer: number[];
  lastFiltered: number;
}

const channelPipelines = EEG_CHANNELS.reduce((acc, channel) => {
  acc[channel] = {
    notch: new NotchFilter(50, SAMPLE_RATE),
    bandpass: new ButterworthBandpass(1, 40, SAMPLE_RATE),
    window: new SlidingWindow(WINDOW_SIZE, HOP_SIZE),
    hopBuffer: [],
    lastFiltered: 0,
  };
  return acc;
}, {} as Record<EEGChannel, ChannelPipelineState>);

const fftAn     = new FFTAnalyzer(WINDOW_SIZE, SAMPLE_RATE);
const bandPow   = new BandPowerExtractor(SAMPLE_RATE, WINDOW_SIZE);

// ✅ NUEVO: Cliente ML para clasificación de estados
const mlClient = new MLServiceClient("ws://localhost:8001/ws");

// ---------------------------------------------------------------------------
// Instancias de infraestructura
// ---------------------------------------------------------------------------

const simulator = new EEGSimulator();
const datasetReplayer = new DatasetReplayer<EEGChannel>(SAMPLE_RATE, EEG_CHANNELS);
const db        = new SessionDB();

type AcquisitionSource = "none" | "simulator" | "dataset";
let activeSource: AcquisitionSource = "none";

function stopAcquisition(): void {
  simulator.stop();
  datasetReplayer.stop();
  activeSource = "none";
}

/**
 * Log de configuración al iniciar.
 * Indica claramente si se usa simulador (desarrollo) o hardware real (producción).
 */
function logStartupConfig(): void {
  console.log("╔════════════════════════════════════════════════════════╗");
  console.log("║  NEUROFEEDBACK EEG SERVER — STARTUP CONFIG             ║");
  console.log("╚════════════════════════════════════════════════════════╝");
  console.log(`  Puerto:           ${PORT}`);
  console.log(`  Sample Rate:      ${SAMPLE_RATE} Hz`);
  console.log(`  Ventana FFT:      ${WINDOW_SIZE} samples (${(WINDOW_SIZE / SAMPLE_RATE * 1000).toFixed(0)} ms)`);
  console.log(`  Hop Size:         ${HOP_SIZE} samples (~${(HOP_SIZE / SAMPLE_RATE * 1000).toFixed(0)} ms entre epochs)`);
  console.log(`  Modo EEG:         ${EEG_MODE === "simulator" ? "🔷 SIMULADOR (desarrollo)" : "🔴 HARDWARE REAL (producción)"}`);
  if (EEG_MODE === "simulator") {
    console.log(`                    $ EEG_MODE=hardware node dist/server.js  ← para hardware real`);
  }
  console.log(`  Base de datos:    SessionDB (TODO: persistencia)`);
  console.log("");
}

// ---------------------------------------------------------------------------
// Estado global de la sesión
// ---------------------------------------------------------------------------

let currentSession : SessionState | null = null;
let lastFilteredSample = 0;
let inspectionChannel: EEGChannel = "Fz";
let loadedDataset: DatasetMetadata | null = null;

/**
 * Resetea todo el estado DSP para una nueva sesión.
 * Llamar al inicio de cada sesión para evitar contaminación de estadísticas.
 */
function resetPipeline(config: SessionConfig): void {
  for (const channel of EEG_CHANNELS) {
    const state = channelPipelines[channel];
    state.notch.reset();
    state.bandpass.reset();
    state.window.reset();
    state.hopBuffer.length = 0;
    state.lastFiltered = 0;
  }
  simulator.reset();
  datasetReplayer.reset();
  console.log("[Pipeline] Reset completado para nueva sesión");
}

// ---------------------------------------------------------------------------
// Métricas de latencia
// ---------------------------------------------------------------------------

/** Acumulador de tiempos de procesamiento de epochs (ms) */
const latencyBuffer: number[] = [];

function recordLatency(ms: number): void {
  latencyBuffer.push(ms);
}

/** Calcula y loggea estadísticas de latencia cada LATENCY_LOG_INTERVAL_MS */
function startLatencyLogger(): ReturnType<typeof setInterval> {
  return setInterval(() => {
    if (latencyBuffer.length === 0) return;

    const n    = latencyBuffer.length;
    const sum  = latencyBuffer.reduce((a, b) => a + b, 0);
    const mean = sum / n;
    const max  = Math.max(...latencyBuffer);
    const min  = Math.min(...latencyBuffer);

    // Percentil 95
    const sorted = [...latencyBuffer].sort((a, b) => a - b);
    const p95    = sorted[Math.floor(n * 0.95)] ?? max;

    const status = mean < 5 ? "✅" : "⚠️ ";
    console.log(
      `[Pipeline latency] ${status} ` +
      `mean=${mean.toFixed(2)}ms  max=${max.toFixed(2)}ms  ` +
      `min=${min.toFixed(2)}ms  p95=${p95.toFixed(2)}ms  ` +
      `epochs=${n}  (objetivo: mean < 5 ms)`
    );

    // Alertar si p95 supera el umbral — puede indicar bloqueo del event loop
    if (p95 >= 5) {
      console.warn(
        "[Pipeline latency] ⚠️  p95 ≥ 5 ms. Posible saturación del event loop. " +
        "Considerar mover FFT a Worker Thread."
      );
    }

    latencyBuffer.length = 0; // limpiar acumulador
  }, LATENCY_LOG_INTERVAL_MS);
}

// ---------------------------------------------------------------------------
// Pipeline principal DSP
// ---------------------------------------------------------------------------

/**
 * Procesa un sample crudo del ADC EEG a través de la cadena completa de
 * filtros y análisis espectral.
 *
 * Solo ejecuta el análisis FFT / bandpower / z-score / feedback cuando
 * SlidingWindow indica que hay un epoch completo listo (cada hopSize muestras).
 * Entre epochs, almacena el sample filtrado para incluirlo en el payload.
 *
 * Manejo de errores:
 *   Cualquier excepción en el pipeline se captura y loggea sin crashear el
 *   proceso — un error en un epoch no debe interrumpir la sesión del paciente.
 */
function clamp01(v: number): number {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

async function processSample(rawByChannel: ChannelSample, wss: WebSocketServer): Promise<void> {
  try {
    // ── Etapa 1: Filtrado por canal ───────────────────────────────────
    let epochReady = false;
    for (const channel of EEG_CHANNELS) {
      const state = channelPipelines[channel];
      const afterNotch = state.notch.process(rawByChannel[channel]);
      const afterBandpass = state.bandpass.process(afterNotch);
      state.lastFiltered = afterBandpass;
      state.hopBuffer.push(afterBandpass);

      const ready = state.window.push(afterBandpass);
      if (channel === "Fz") epochReady = ready;
    }

    lastFilteredSample = channelPipelines.Fz.lastFiltered;
    if (!epochReady) return;

    // ── Etapa 2: Análisis espectral (solo cuando hay epoch completo) ──
    const t0 = Date.now();

    const bandsByChannel: Record<EEGChannel, BandPowers> = {} as Record<EEGChannel, BandPowers>;
    const magnitudesByChannel: Record<EEGChannel, Float32Array> = {} as Record<EEGChannel, Float32Array>;
    const windowsByChannel: Partial<Record<EEGChannel, Float32Array>> = {};

    for (const channel of EEG_CHANNELS) {
      const state = channelPipelines[channel];
      const rawWindow  = state.window.getWindow();
      const hannWindow = state.window.applyHannWindow(rawWindow);
      const magnitudes = fftAn.analyze(hannWindow);
      magnitudesByChannel[channel] = magnitudes;
      bandsByChannel[channel] = bandPow.extract(magnitudes);
      windowsByChannel[channel] = rawWindow;
    }

    const mlWindow = ML_CHANNELS.map((channel) => Array.from(windowsByChannel[channel]!));
    const mlBandPowers = Object.fromEntries(
      ML_CHANNELS.map((channel) => [channel, bandsByChannel[channel]])
    ) as unknown as Record<string, Record<string, number>>;

    const fzBands = bandsByChannel.Fz;
    const thetaBeta = bandPow.computeThetaBetaRatio(fzBands);

    const thetaFz = fzBands.theta;
    const thetaLateral = (bandsByChannel.F3.theta + bandsByChannel.F4.theta) / 2;
    const frontalSpecificity = thetaFz / Math.max(thetaLateral, 1e-12);
    const frontalThreshold = DEFAULT_FRONTAL_SPECIFICITY_THRESHOLD;
    const frontalSpecificityValid = frontalSpecificity >= frontalThreshold;

    const fp1Peak = Math.max(...channelPipelines.Fp1.hopBuffer.map(v => Math.abs(v)), 0);
    const fp2Peak = Math.max(...channelPipelines.Fp2.hopBuffer.map(v => Math.abs(v)), 0);
    const artifactDetected = fp1Peak > ARTIFACT_THRESHOLD_UV || fp2Peak > ARTIFACT_THRESHOLD_UV;

    const thetaPeaks = EEG_CHANNELS.map((channel) => bandsByChannel[channel].theta);
    const thetaMax = Math.max(...thetaPeaks, 1e-12);
    const topographyTheta = Object.fromEntries(
      EEG_CHANNELS.map((channel) => [channel, clamp01(bandsByChannel[channel].theta / thetaMax)])
    ) as Record<EEGChannel, number>;

    // ── Etapa 3: Clasificación via ML Service ────────────────────────
    // ✅ NUEVO: Llamar al clasificador (retorna null si el servicio no está disponible)
    const mlResult = await mlClient.processWindow({
      eeg_window: mlWindow,
      band_powers_per_channel: mlBandPowers,
      frontal_specificity: frontalSpecificity,
    });

    const pipelineMs = Date.now() - t0;
    recordLatency(pipelineMs);

    // ── Etapa 4: Broadcast al frontend ───────────────────────────────
    // Capturar y vaciar el hop buffer acumulado (64 muestras) para que el
    // osciloscopio del frontend dibuje a 250 sps reales en vez de 4 sps.
    const hopSamples = channelPipelines.Fz.hopBuffer.splice(0);
    const inspectionHopSamples = channelPipelines[inspectionChannel].hopBuffer.splice(0);
    const inspectionMagnitudes = Array.from(magnitudesByChannel[inspectionChannel]).slice(0, 96);
    for (const channel of EEG_CHANNELS) {
      if (channel !== "Fz" && channel !== inspectionChannel) {
        channelPipelines[channel].hopBuffer.length = 0;
      }
    }

    const payload: FeedbackPayload = {
      type            : "eeg_data",
      timestamp       : Date.now(),
      filteredSample  : lastFilteredSample,
      filteredSamples : hopSamples,
      frontalSpecificity,
      frontalSpecificityValid,
      artifactDetected,
      topographyTheta,
      inspection: {
        channel: inspectionChannel,
        filteredSamples: inspectionHopSamples,
        fftMagnitudes: inspectionMagnitudes,
        bandPowers: bandsByChannel[inspectionChannel],
      },
      bandPowers      : fzBands,
      thetaBeta,
      // ✅ NUEVO: Predicción de estado del clasificador
      state_prediction: mlResult?.classifier_prediction ?? null,
      connectivity    : mlResult ?? null,
      pipelineMs,
    };

    broadcast(wss, payload);

  } catch (err) {
    // Error aislado — loggear y continuar; no debe detener la sesión
    console.error("[Pipeline] Error en processSample:", err);
  }
}

// ---------------------------------------------------------------------------
// Utilidades WebSocket
// ---------------------------------------------------------------------------

/** Envía un objeto JSON a todos los clientes WS en estado OPEN */
function broadcast(wss: WebSocketServer, data: object): void {
  const json = JSON.stringify(data);
  for (const client of wss.clients) {
    if (client.readyState === WebSocket.OPEN) {
      client.send(json, (err) => {
        if (err) console.error("[WS] Error en send:", err.message);
      });
    }
  }
}

/** Envía un objeto JSON a un cliente WS específico */
function send(ws: WebSocket, data: object): void {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data), (err) => {
      if (err) console.error("[WS] Error en send individual:", err.message);
    });
  }
}

// ---------------------------------------------------------------------------
// Manejadores de mensajes entrantes
// ---------------------------------------------------------------------------

function handleMessage(ws: WebSocket, wss: WebSocketServer, raw: string): void {
  let msg: IncomingMessage;

  try {
    msg = JSON.parse(raw) as IncomingMessage;
  } catch {
    send(ws, { type: "error", message: "JSON inválido" });
    return;
  }

  switch (msg.type) {

    // ── start_session ────────────────────────────────────────────────────
    case "start_session": {
      if (currentSession) {
        // Detener sesión anterior sin error — cliente reconecta
        stopAcquisition();
        console.log(`[Session] Sesión anterior ${currentSession.id} reemplazada.`);
      }

      const config     = (msg as { type: "start_session"; payload: SessionConfig }).payload ?? {};
      const sessionId  = `session_${Date.now()}`;

      currentSession = {
        id        : sessionId,
        startedAt : Date.now(),
        tranceMode: false,
        config,
      };

      resetPipeline(config);

      // En modo hardware, solo aceptar fuente real (dataset cargado o driver real futuro).
      if (EEG_MODE === "hardware") {
        if (!loadedDataset || !datasetReplayer.isLoaded()) {
          currentSession = null;
          send(ws, {
            type: "error",
            message:
              "EEG_MODE=hardware activo: falta dataset real cargado. " +
              "Usa CARGAR DATASET y luego inicia sesión.",
          });
          return;
        }

        datasetReplayer.start((sample) => processSample(sample, wss));
        activeSource = "dataset";
      } else {
        // Solo permitido en modo de desarrollo explícito.
        simulator.start((sample) => processSample(sample, wss));
        activeSource = "simulator";
      }

      send(ws, { type: "session_started", sessionId });
      console.log(`[Session] ▶  Sesión iniciada: ${sessionId} (${activeSource})`);
      break;
    }

    // ── stop_session ─────────────────────────────────────────────────────
    case "stop_session": {
      if (!currentSession) {
        send(ws, { type: "error", message: "No hay sesión activa" });
        return;
      }

      stopAcquisition();
      const stoppedId  = currentSession.id;
      const durationMs = Date.now() - currentSession.startedAt;
      currentSession   = null;

      broadcast(wss, { type: "session_stopped" });
      console.log(
        `[Session] ■  Sesión detenida: ${stoppedId}  ` +
        `duración: ${(durationMs / 1000).toFixed(1)}s`
      );
      break;
    }

    // ── set_trance_mode ──────────────────────────────────────────────────
    case "set_trance_mode": {
      const { enabled } = (msg as { type: "set_trance_mode"; payload: { enabled: boolean }}).payload;

      if (!currentSession) {
        send(ws, { type: "error", message: "No hay sesión activa para set_trance_mode" });
        return;
      }

      currentSession.tranceMode = enabled;
      if (activeSource === "simulator") {
        simulator.setTranceMode(enabled);
      }
      send(ws, { type: "trance_mode_set", enabled });
      console.log(`[Session] Trance mode: ${enabled ? "ON 🌀" : "OFF"}`);
      break;
    }

    // ── set_inspection_channel ──────────────────────────────────────────
    case "set_inspection_channel": {
      const { channel } = (msg as { type: "set_inspection_channel"; payload: { channel: EEGChannel } }).payload;

      if (!EEG_CHANNELS.includes(channel)) {
        send(ws, { type: "error", message: `Canal de inspección inválido: ${channel}` });
        return;
      }

      inspectionChannel = channel;
      send(ws, { type: "inspection_channel_set", channel });
      console.log(`[Session] Inspection channel: ${channel}`);
      break;
    }

    // ── load_dataset ───────────────────────────────────────────────────
    case "load_dataset": {
      const datasetPath = (msg as { type: "load_dataset"; payload: { path: string } }).payload?.path;
      if (!datasetPath || typeof datasetPath !== "string") {
        send(ws, { type: "error", message: "load_dataset: falta path" });
        return;
      }

      parseDatasetMetadata(datasetPath)
        .then(async (meta) => {
          await datasetReplayer.load(datasetPath);
          loadedDataset = meta;
          send(ws, { type: "dataset_loaded", dataset: meta });
          console.log(
            `[Dataset] loaded ${meta.format.toUpperCase()} ` +
            `${meta.channels.length}ch ${meta.sampleRate}Hz ${meta.durationSec.toFixed(1)}s`
          );
        })
        .catch((err: Error) => {
          send(ws, { type: "error", message: `load_dataset: ${err.message}` });
        });
      break;
    }

    // ── ping ─────────────────────────────────────────────────────────────
    case "ping": {
      send(ws, { type: "pong", timestamp: Date.now() });
      break;
    }

    // ── submit_subjective ────────────────────────────────────────────────
    case "submit_subjective": {
      const measure = (msg as { type: "submit_subjective"; payload: SubjectiveMeasure }).payload;

      if (!measure?.sessionId) {
        send(ws, { type: "error", message: "submit_subjective: falta sessionId" });
        return;
      }

      // Inserción asíncrona — no bloqueamos el event loop
      db.insertSubjectiveMeasure(measure)
        .then(() => {
          send(ws, { type: "subjective_saved", sessionId: measure.sessionId });
          console.log(`[DB] Medida subjetiva guardada: ${measure.label}=${measure.value}`);
        })
        .catch((err: Error) => {
          console.error("[DB] Error insertSubjectiveMeasure:", err.message);
          send(ws, {
            type    : "error",
            message : `Error al guardar medida subjetiva: ${err.message}`,
          });
        });
      break;
    }

    default: {
      send(ws, {
        type    : "error",
        message : `Tipo de mensaje desconocido: ${(msg as { type: string }).type}`,
      });
    }
  }
}

// ---------------------------------------------------------------------------
// Bootstrap del servidor
// ---------------------------------------------------------------------------

/** Crea el servidor HTTP (health-check + upgrade WS) */
function createHttpServer(): http.Server {
  return http.createServer((req, res) => {
    if (req.url === "/health" && req.method === "GET") {
      const health = {
        status    : "ok",
        uptime    : process.uptime(),
        session   : currentSession
          ? { id: currentSession.id, tranceMode: currentSession.tranceMode }
          : null,
        timestamp : new Date().toISOString(),
      };
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(health));
    } else {
      res.writeHead(404);
      res.end("Not Found");
    }
  });
}

function main(): void {
  const httpServer = createHttpServer();

  // ✅ NUEVO: Conectar al servicio ML
  mlClient.connect();
  console.log("[ML-Client] Iniciando conexión con ML Service en ws://localhost:8001/ws");

  // ── WebSocket Server ───────────────────────────────────────────────────
  const wss = new WebSocketServer({ server: httpServer });

  wss.on("connection", (ws: WebSocket, req) => {
    const ip = req.socket.remoteAddress ?? "unknown";
    console.log(`[WS] Cliente conectado: ${ip}  (total: ${wss.clients.size})`);

    // Enviar estado inicial al cliente recién conectado
    send(ws, {
      type      : "server_hello",
      version   : "1.0.0",
      sampleRate: SAMPLE_RATE,
      windowSize: WINDOW_SIZE,
      hopSize   : HOP_SIZE,
      dataset   : loadedDataset,
      session   : currentSession
        ? { id: currentSession.id, tranceMode: currentSession.tranceMode }
        : null,
    });

    ws.on("message", (data) => {
      handleMessage(ws, wss, data.toString());
    });

    ws.on("close", (code, reason) => {
      console.log(
        `[WS] Cliente desconectado: ${ip}  ` +
        `code=${code}  reason=${reason.toString() || "—"}  ` +
        `(restantes: ${wss.clients.size})`
      );
    });

    ws.on("error", (err) => {
      console.error(`[WS] Error en cliente ${ip}:`, err.message);
    });
  });

  wss.on("error", (err) => {
    console.error("[WSS] Error del servidor WebSocket:", err.message);
  });

  // ── Logger de latencia ────────────────────────────────────────────────
  const latencyTimer = startLatencyLogger();

  // ── Arrancar servidor HTTP ────────────────────────────────────────────
  httpServer.listen(PORT, () => {
    logStartupConfig();
    console.log("  ✅ Servidor activo — aceptando conexiones WS y HTTP");
    console.log("");
  });

  // ── Graceful shutdown ─────────────────────────────────────────────────
  const shutdown = (signal: string) => {
    console.log(`\n[Server] ${signal} recibido. Cerrando limpiamente…`);
    clearInterval(latencyTimer);
    stopAcquisition();

    wss.clients.forEach((client) => client.terminate());
    wss.close(() => {
      httpServer.close(() => {
        console.log("[Server] Servidor cerrado correctamente. Bye 👋");
        process.exit(0);
      });
    });
  };

  process.on("SIGTERM", () => shutdown("SIGTERM"));
  process.on("SIGINT",  () => shutdown("SIGINT"));

  // Prevenir crash por promesa rechazada no manejada
  process.on("unhandledRejection", (reason) => {
    console.error("[Process] unhandledRejection:", reason);
  });

  process.on("uncaughtException", (err) => {
    console.error("[Process] uncaughtException:", err.message, err.stack);
    // No salir — el servidor sigue en pie para la sesión del paciente
  });
}

main();
