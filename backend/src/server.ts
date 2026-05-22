/**
 * server.ts  —  backend/src/
 *
 * Orquestador principal del sistema de neurofeedback EEG.
 *
 * Responsabilidades:
 *   1. Servidor HTTP (health-check) + WebSocket server en puerto 8080.
 *   2. Gestión del ciclo de vida de sesión (start / stop / reset).
 *   3. Ejecución del pipeline DSP por cada sample EEG real cargado desde dataset.
 *   4. Distribución del FeedbackPayload a todos los clientes WS conectados.
 *   5. Medición continua de latencia del pipeline (objetivo: < 5 ms).
 *
 * ── Arquitectura del pipeline (por sample) ───────────────────────────────
 *
 *   DatasetReplayer (250 sps)
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
 *     { type: 'load_dataset',      payload: { path: string }   }
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
import fs from "fs/promises";
import path from "path";
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

/** Intervalo del log de latencia (ms) */
const LATENCY_LOG_INTERVAL_MS = 5_000;

/** Mínimo de samples del z-score antes de enviar comandos de feedback */
const ZSCORE_WARMUP_SAMPLES = 30;
const DEFAULT_FRONTAL_SPECIFICITY_THRESHOLD = 1.5;
const ARTIFACT_THRESHOLD_UV = 100;
const PREDICTION_SMOOTHING_WINDOWS = Number(process.env.PREDICTION_SMOOTHING_WINDOWS ?? 5);
const PREDICTION_SMOOTHING_MIN_HISTORY = 3;
const PREDICTION_SMOOTHING_MIN_MAJORITY = 0.6;

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
  playback        : { positionSec: number; durationSec: number } | null;
  pipelineStageMs?: {
    referenceAndFilterMs: number;
    spectralAndFeatureMs: number;
    totalMs: number;
  };
  pipelineMs      : number;        // tiempo de procesamiento del epoch [ms]
}

type IncomingMessage =
  | { type: "start_session";    payload: SessionConfig                }
  | { type: "stop_session"                                            }
  | { type: "set_inspection_channel"; payload: { channel: EEGChannel } }
  | { type: "load_dataset"; payload: { path: string }                 }
  | { type: "set_playback_position"; payload: { seconds: number }     }
  | { type: "ping"                                                    }
  | { type: "submit_subjective"; payload: SubjectiveMeasure          };

// ---------------------------------------------------------------------------
// Estado de sesión
// ---------------------------------------------------------------------------

interface SessionState {
  id         : string;
  startedAt  : number;
  config     : SessionConfig;
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
let lastMLResult: MLServiceResult | null = null;
let mlRequestInFlight = false;
const predictionLabelHistory: Array<ClassifierPrediction["predicted_label"]> = [];

// ---------------------------------------------------------------------------
// Instancias de infraestructura
// ---------------------------------------------------------------------------

const datasetReplayer = new DatasetReplayer<EEGChannel>(SAMPLE_RATE, EEG_CHANNELS);
const db        = new SessionDB();

type AcquisitionSource = "none" | "dataset";
let activeSource: AcquisitionSource = "none";

function stopAcquisition(): void {
  datasetReplayer.stop();
  activeSource = "none";
  mlRequestInFlight = false;
  lastMLResult = null;
  predictionLabelHistory.length = 0;
}

function smoothPrediction(
  prediction: ClassifierPrediction,
): ClassifierPrediction {
  const windowSize = Math.max(1, PREDICTION_SMOOTHING_WINDOWS);
  predictionLabelHistory.push(prediction.predicted_label);
  if (predictionLabelHistory.length > windowSize) {
    predictionLabelHistory.shift();
  }

  // No suavizar hasta tener historial minimo: evita pegarse al primer estado.
  if (predictionLabelHistory.length < PREDICTION_SMOOTHING_MIN_HISTORY) {
    return prediction;
  }

  const counts = new Map<ClassifierPrediction["predicted_label"], number>();
  for (const label of predictionLabelHistory) {
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  let winner = prediction.predicted_label;
  let winnerCount = counts.get(winner) ?? 0;
  for (const [label, count] of counts.entries()) {
    if (count > winnerCount) {
      winner = label;
      winnerCount = count;
    }
  }

  const winnerRatio = winnerCount / predictionLabelHistory.length;
  if (winnerRatio < PREDICTION_SMOOTHING_MIN_MAJORITY) {
    return prediction;
  }

  if (winner === prediction.predicted_label) {
    return prediction;
  }

  const classToLabel: Record<number, ClassifierPrediction["predicted_label"]> = {
    0: "awake",
    1: "induction",
    2: "trance",
  };
  const labelToClass: Record<string, number> = {
    awake: 0,
    induction: 1,
    trance: 2,
    uncertain: -1,
  };

  const smoothed = {
    ...prediction,
    predicted_label: winner,
    predicted_class: labelToClass[winner] ?? prediction.predicted_class,
    method: "ml_fused" as const,
  };

  if (smoothed.predicted_class >= 0) {
    smoothed.class_probabilities = {
      awake: winner === classToLabel[0] ? 1 : 0,
      induction: winner === classToLabel[1] ? 1 : 0,
      trance: winner === classToLabel[2] ? 1 : 0,
    };
  }
  return smoothed;
}

/**
 * Log de configuración al iniciar.
 * Adquisición estricta desde dataset real (EDF/CSV), sin simulación.
 */
function logStartupConfig(): void {
  console.log("╔════════════════════════════════════════════════════════╗");
  console.log("║  NEUROFEEDBACK EEG SERVER — STARTUP CONFIG             ║");
  console.log("╚════════════════════════════════════════════════════════╝");
  console.log(`  Puerto:           ${PORT}`);
  console.log(`  Sample Rate:      ${SAMPLE_RATE} Hz`);
  console.log(`  Ventana FFT:      ${WINDOW_SIZE} samples (${(WINDOW_SIZE / SAMPLE_RATE * 1000).toFixed(0)} ms)`);
  console.log(`  Hop Size:         ${HOP_SIZE} samples (~${(HOP_SIZE / SAMPLE_RATE * 1000).toFixed(0)} ms entre epochs)`);
  console.log("  Modo EEG:         🔴 DATASET REAL (sin simulación)");
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
    const totalT0 = Date.now();
    const referenceAndFilterT0 = Date.now();
    const averageReference = ML_CHANNELS.reduce(
      (sum, channel) => sum + rawByChannel[channel],
      0
    ) / ML_CHANNELS.length;

    let epochReady = false;
    for (const channel of EEG_CHANNELS) {
      const state = channelPipelines[channel];
      const referencedSample = rawByChannel[channel] - averageReference;

      const afterNotch = state.notch.process(referencedSample);
      const afterBandpass = state.bandpass.process(afterNotch);
      state.lastFiltered = afterBandpass;
      state.hopBuffer.push(afterBandpass);

      const ready = state.window.push(afterBandpass);
      if (channel === "Fz") epochReady = ready;
    }

    lastFilteredSample = channelPipelines.Fz.lastFiltered;
    if (!epochReady) return;
    const referenceAndFilterMs = Date.now() - referenceAndFilterT0;

    // ── Etapa 2: Análisis espectral (solo cuando hay epoch completo) ──
    const spectralAndFeatureT0 = Date.now();

    const bandsByChannel: Record<EEGChannel, BandPowers> = {} as Record<EEGChannel, BandPowers>;
    const magnitudesByChannel: Record<EEGChannel, Float32Array> = {} as Record<EEGChannel, Float32Array>;
    const filteredWindowsByChannel: Partial<Record<EEGChannel, Float32Array>> = {};

    for (const channel of EEG_CHANNELS) {
      const state = channelPipelines[channel];
      const filteredWindow = state.window.getWindow();
      const hannWindow = state.window.applyHannWindow(filteredWindow);
      const magnitudes = fftAn.analyze(hannWindow);
      magnitudesByChannel[channel] = magnitudes;
      bandsByChannel[channel] = bandPow.extract(magnitudes);
      filteredWindowsByChannel[channel] = filteredWindow;
    }

    const mlWindow = ML_CHANNELS.map((channel) => Array.from(filteredWindowsByChannel[channel]!));
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

    // ── Etapa 3: Clasificación via ML Service (no bloqueante) ────────
    // Evita bloquear el loop DSP cuando el servicio ML responde lento.
    if (!mlRequestInFlight) {
      mlRequestInFlight = true;
      void mlClient.processWindow({
        eeg_window: mlWindow,
        band_powers_per_channel: mlBandPowers,
        frontal_specificity: frontalSpecificity,
        model_profile_mode: currentSession?.config?.modelProfileMode ?? "auto",
      })
        .then((result) => {
          if (result) {
            lastMLResult = result;
            const timings = result.stage_timings_ms;
            if (timings) {
              console.log(
                `[ML timings] preprocess=${timings.preprocess_ms.toFixed(2)}ms ` +
                `coherence=${timings.coherence_ms.toFixed(2)}ms ` +
                `plv=${timings.plv_ms.toFixed(2)}ms ` +
                `features=${timings.feature_ms.toFixed(2)}ms ` +
                `model=${timings.model_ms.toFixed(2)}ms ` +
                `total=${result.processing_ms.toFixed(2)}ms`
              );
            }
          }
        })
        .catch((err) => {
          console.warn(`[ML-Client] processWindow falló: ${(err as Error).message}`);
        })
        .finally(() => {
          mlRequestInFlight = false;
        });
    }

    const spectralAndFeatureMs = Date.now() - spectralAndFeatureT0;
    const pipelineMs = Date.now() - totalT0;
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
      // Predicción suavizada temporalmente por mayoría deslizante.
      state_prediction: lastMLResult?.classifier_prediction
        ? smoothPrediction(lastMLResult.classifier_prediction)
        : null,
      connectivity    : lastMLResult,
      playback        : activeSource === "dataset" ? datasetReplayer.getPlaybackInfo() : null,
      pipelineStageMs : {
        referenceAndFilterMs,
        spectralAndFeatureMs,
        totalMs: pipelineMs,
      },
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
        config,
      };

      resetPipeline(config);

      if (!loadedDataset || !datasetReplayer.isLoaded()) {
        currentSession = null;
        send(ws, {
          type: "error",
          message: "Falta dataset real cargado. Usa CARGAR DATASET y luego inicia sesión.",
        });
        return;
      }

      datasetReplayer.start((sample) => processSample(sample, wss));
      activeSource = "dataset";

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

      // Cargar un nuevo dataset debe dejar la sesión anterior completamente limpia.
      // Si queda el cursor viejo o un intervalo activo, la UI parece congelarse.
      if (currentSession) {
        stopAcquisition();
        currentSession = null;
        broadcast(wss, { type: "session_stopped" });
      } else {
        stopAcquisition();
      }

      datasetReplayer.reset();

      parseDatasetMetadata(datasetPath)
        .then(async (meta) => {
          await datasetReplayer.load(datasetPath);
          loadedDataset = meta;
          send(ws, {
            type: "dataset_loaded",
            dataset: meta,
            playback: datasetReplayer.getPlaybackInfo(),
          });
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

    // ── set_playback_position ──────────────────────────────────────────
    case "set_playback_position": {
      const seconds = (msg as { type: "set_playback_position"; payload: { seconds: number } }).payload?.seconds;
      if (!datasetReplayer.isLoaded()) {
        send(ws, { type: "error", message: "set_playback_position: no hay dataset cargado" });
        return;
      }

      try {
        const playback = datasetReplayer.seekToSeconds(Number(seconds));
        send(ws, { type: "playback_position_set", ...playback });
      } catch (err) {
        send(ws, { type: "error", message: `set_playback_position: ${(err as Error).message}` });
      }
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
async function listDatasetsFromDataDir(): Promise<string[]> {
  const root = path.resolve(process.cwd(), "..", "data");

  async function walk(dir: string): Promise<string[]> {
    let entries: Array<{ name: string; isDirectory: () => boolean }> = [];
    try {
      const rawEntries = await fs.readdir(dir, { withFileTypes: true });
      entries = rawEntries.map((entry) => ({
        name: String(entry.name),
        isDirectory: () => entry.isDirectory(),
      }));
    } catch {
      return [];
    }

    const out: string[] = [];
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        out.push(...(await walk(full)));
        continue;
      }

      const ext = path.extname(entry.name).toLowerCase();
      if (ext === ".edf" || ext === ".csv") {
        out.push(full);
      }
    }
    return out;
  }

  const files = await walk(root);
  return files.sort((a, b) => a.localeCompare(b));
}

function createHttpServer(): http.Server {
  return http.createServer(async (req, res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET,OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    if (req.url === "/health" && req.method === "GET") {
      const health = {
        status    : "ok",
        uptime    : process.uptime(),
        session   : currentSession
          ? { id: currentSession.id }
          : null,
        timestamp : new Date().toISOString(),
      };
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(health));
    } else if (req.url === "/datasets" && req.method === "GET") {
      try {
        const datasets = await listDatasetsFromDataDir();
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ datasets }));
      } catch (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            error: `No se pudo listar datasets: ${(err as Error).message}`,
          })
        );
      }
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
      playback  : datasetReplayer.getPlaybackInfo(),
      session   : currentSession
        ? { id: currentSession.id }
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
