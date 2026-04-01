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
 *     { type: 'ping'                                          }
 *     { type: 'submit_subjective', payload: SubjectiveMeasure }
 *
 *   SALIENTE (server → frontend):
 *     { type: 'eeg_data',          ...FeedbackPayload         }
 *     { type: 'pong',              timestamp: number          }
 *     { type: 'session_started',   sessionId: string          }
 *     { type: 'session_stopped'                               }
 *     { type: 'subjective_saved',  sessionId: string          }
 *     { type: 'error',             message: string            }
 */

import http     from "http";
import { WebSocketServer, WebSocket } from "ws";

// ── Módulos DSP ──────────────────────────────────────────────────────────────
import { NotchFilter }         from "./filters/notch-filter";
import { ButterworthBandpass } from "./filters/butterworth-filter";
import { SlidingWindow }       from "./dsp/sliding-window";
import { FFTAnalyzer }         from "./dsp/fft-analyzer";
import { BandPowerExtractor }  from "./dsp/band-power";

// ── Módulos adaptativos ──────────────────────────────────────────────────────
import { RunningZScore }   from "./adaptive/running-zscore";
import { FeedbackEngine }  from "./adaptive/feedback-engine";

// ── Tipos compartidos ────────────────────────────────────────────────────────
import type { BandPowers, ThetaBetaResult }  from "./dsp/band-power";
import type { ZScoreResult }                 from "./adaptive/running-zscore";
import type { FeedbackCommand, SessionConfig } from "./adaptive/feedback-engine";

// ---------------------------------------------------------------------------
// Constantes de configuración
// ---------------------------------------------------------------------------

const PORT        = Number(process.env.PORT ?? 8080);
const SAMPLE_RATE = 250;   // Hz — OpenBCI Cyton default
const WINDOW_SIZE = 256;   // samples — 1.024 s de epoch
const HOP_SIZE    = 64;    // samples — 75 % de overlap → ~4 payloads/s

/** Intervalo del log de latencia (ms) */
const LATENCY_LOG_INTERVAL_MS = 5_000;

/** Mínimo de samples del z-score antes de enviar comandos de feedback */
const ZSCORE_WARMUP_SAMPLES = 30;

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
  type           : "eeg_data";
  timestamp      : number;
  filteredSample : number;       // último sample filtrado [µV]
  bandPowers     : BandPowers;
  thetaBeta      : ThetaBetaResult;
  zScore         : ZScoreResult;
  command        : FeedbackCommand;
  pipelineMs     : number;       // tiempo de procesamiento del epoch [ms]
}

type IncomingMessage =
  | { type: "start_session";    payload: SessionConfig                }
  | { type: "stop_session"                                            }
  | { type: "set_trance_mode";  payload: { enabled: boolean }        }
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

  start(onSample: (sample: number) => void): void {
    if (this.intervalId) return;
    const intervalMs = 1000 / SAMPLE_RATE; // 4 ms entre samples a 250 Hz

    this.intervalId = setInterval(() => {
      const sample = this._generateSample();
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
  }

  private _generateSample(): number {
    const fs = SAMPLE_RATE;
    const t  = this.t / fs;

    if (this.tranceMode) {
      // Señal theta-dominante: 6 Hz fuerte + 10 Hz alfa suave + ruido
      return (
        15 * Math.sin(2 * Math.PI * 6  * t) +   // theta fuerte
         5 * Math.sin(2 * Math.PI * 10 * t) +   // alfa suave
         2 * Math.sin(2 * Math.PI * 50 * t) +   // interferencia 50 Hz (para demostrar notch)
         this._noise(3)
      );
    } else {
      // Señal beta-dominante: estado alerta / activo
      return (
         5 * Math.sin(2 * Math.PI * 20 * t) +   // beta medio
         4 * Math.sin(2 * Math.PI * 10 * t) +   // alfa moderado
         3 * Math.sin(2 * Math.PI * 6  * t) +   // theta bajo
         2 * Math.sin(2 * Math.PI * 50 * t) +   // interferencia 50 Hz
         this._noise(4)
      );
    }
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

const notch     = new NotchFilter(50, SAMPLE_RATE);          // elimina 50 Hz
const bandpass  = new ButterworthBandpass(1, 40, SAMPLE_RATE); // pasa 1–40 Hz
const window_   = new SlidingWindow(WINDOW_SIZE, HOP_SIZE);
const fftAn     = new FFTAnalyzer(WINDOW_SIZE, SAMPLE_RATE);
const bandPow   = new BandPowerExtractor(SAMPLE_RATE, WINDOW_SIZE);
const zScore    = new RunningZScore(3);                      // ventana de 3 min

// FeedbackEngine — se recrea con cada SessionConfig en start_session
let feedbackEng = new FeedbackEngine();

// ---------------------------------------------------------------------------
// Instancias de infraestructura
// ---------------------------------------------------------------------------

const simulator = new EEGSimulator();
const db        = new SessionDB();

// ---------------------------------------------------------------------------
// Estado global de la sesión
// ---------------------------------------------------------------------------

let currentSession : SessionState | null = null;
let lastFilteredSample = 0;

/**
 * Resetea todo el estado DSP para una nueva sesión.
 * Llamar al inicio de cada sesión para evitar contaminación de estadísticas.
 */
function resetPipeline(config: SessionConfig): void {
  notch.reset();
  bandpass.reset();
  window_.reset();
  zScore.reset();
  feedbackEng = new FeedbackEngine(config);
  simulator.reset();
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
function processSample(rawSample: number, wss: WebSocketServer): void {
  try {
    // ── Etapa 1: Filtrado ──────────────────────────────────────────────
    const afterNotch    = notch.process(rawSample);
    const afterBandpass = bandpass.process(afterNotch);
    lastFilteredSample  = afterBandpass;

    // ── Etapa 2: Buffer deslizante ────────────────────────────────────
    const epochReady = window_.push(afterBandpass);
    if (!epochReady) return; // aún no hay suficientes muestras

    // ── Etapa 3: Análisis espectral (solo cuando hay epoch completo) ──
    const t0 = Date.now();

    const rawWindow    = window_.getWindow();
    const hannWindow   = window_.applyHannWindow(rawWindow);
    const magnitudes   = fftAn.analyze(hannWindow);
    const bands        = bandPow.extract(magnitudes);
    const thetaBeta    = bandPow.computeThetaBetaRatio(bands);

    // ── Etapa 4: Normalización adaptativa ────────────────────────────
    const zResult      = zScore.push(thetaBeta.ratio);

    // ── Etapa 5: Motor de feedback ────────────────────────────────────
    // No enviar comandos hasta que el z-score tenga suficientes muestras
    // para ser estadísticamente significativo (Welford n ≥ 30).
    const command = feedbackEng.computeCommand(zResult, thetaBeta);

    const pipelineMs = Date.now() - t0;
    recordLatency(pipelineMs);

    // ── Etapa 6: Broadcast al frontend ───────────────────────────────
    const payload: FeedbackPayload = {
      type           : "eeg_data",
      timestamp      : Date.now(),
      filteredSample : lastFilteredSample,
      bandPowers     : bands,
      thetaBeta,
      zScore         : zResult,
      command,
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
        simulator.stop();
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

      // Iniciar generación de samples — cada sample dispara processSample
      simulator.start((sample) => processSample(sample, wss));

      send(ws, { type: "session_started", sessionId });
      console.log(`[Session] ▶  Sesión iniciada: ${sessionId}`);
      break;
    }

    // ── stop_session ─────────────────────────────────────────────────────
    case "stop_session": {
      if (!currentSession) {
        send(ws, { type: "error", message: "No hay sesión activa" });
        return;
      }

      simulator.stop();
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
      simulator.setTranceMode(enabled);
      send(ws, { type: "trance_mode_set", enabled });
      console.log(`[Session] Trance mode: ${enabled ? "ON 🌀" : "OFF"}`);
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
    console.log("╔══════════════════════════════════════════════════════╗");
    console.log("║         EEG Neurofeedback Server  v1.0.0             ║");
    console.log("╠══════════════════════════════════════════════════════╣");
    console.log(`║  WebSocket  ws://localhost:${PORT}                    ║`);
    console.log(`║  Health     http://localhost:${PORT}/health            ║`);
    console.log(`║  Pipeline   ${SAMPLE_RATE} sps | win=${WINDOW_SIZE} | hop=${HOP_SIZE}          ║`);
    console.log("╚══════════════════════════════════════════════════════╝");
  });

  // ── Graceful shutdown ─────────────────────────────────────────────────
  const shutdown = (signal: string) => {
    console.log(`\n[Server] ${signal} recibido. Cerrando limpiamente…`);
    clearInterval(latencyTimer);
    simulator.stop();

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
