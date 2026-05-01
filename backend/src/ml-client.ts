/**
 * ml-client.ts  —  backend/src/
 *
 * Cliente WebSocket que conecta el servidor Node.js con el servicio Python
 * de análisis de conectividad EEG (ml-service/server.py en puerto 8001).
 *
 * Responsabilidades:
 *   - Mantener una conexión WS persistente con el servicio Python.
 *   - Enviar ventanas EEG de 2 s cada hop (~256 ms) con las band powers
 *     y el frontal specificity index ya calculados por el pipeline existente.
 *   - Retornar el resultado (coherencia, PLV, predicción de estado)
 *     para incluirlo en el FeedbackPayload al frontend React.
 *   - Reconectar automáticamente si el servicio Python no está disponible.
 *
 * Uso en server.ts:
 *
 *   import { MLServiceClient } from "./ml-client";
 *
 *   const mlClient = new MLServiceClient("ws://localhost:8001/ws");
 *   mlClient.connect();
 *
 *   // Dentro del callback de hop DSP:
 *   const mlResult = await mlClient.processWindow({
 *     eeg_window:              multiChannelWindowBuffer, // number[][] [11][500]
 *     band_powers_per_channel: allChannelBandPowers,     // Record<string, BandPowers>
 *     frontal_specificity:     frontalSpecificity,       // number
 *   });
 *
 *   if (mlResult) {
 *     // Incluir en FeedbackPayload:
 *     payload.connectivity = mlResult;
 *   }
 */

import { WebSocket } from "ws";

// ── Tipos del resultado del servicio Python ──────────────────────────────────

export interface ConnectivityFeatures {
  coh_Fz_Pz: number;
  coh_F3_F4: number;
  coh_C3_C4: number;
  coh_Fz_Cz: number;
  coh_O1_O2: number;
  plv_Fz_Pz: number;
  plv_F3_F4: number;
  plv_C3_C4: number;
  plv_Fz_Cz: number;
  plv_O1_O2: number;
}

export interface ClassifierPrediction {
  predicted_class:   number;          // 0=awake, 1=induction, 2=trance
  predicted_label:   "awake" | "induction" | "trance";
  confidence:        number;          // [0, 1]
  class_probabilities: {
    awake:     number;
    induction: number;
    trance:    number;
  };
  is_confident: boolean;              // confidence >= 0.50
  method:       "ml_ensemble" | "heuristic_fallback";
}

export interface MLServiceResult {
  coherence_matrix:      number[][];
  plv_matrix:            number[][];
  channel_labels:        string[];
  connectivity_features: ConnectivityFeatures;
  classifier_prediction: ClassifierPrediction;
  feature_vector:        number[];    // 15 features (debug/logging)
  processing_ms:         number;
}

export interface ProcessWindowPayload {
  eeg_window:              number[][];                  // [11][500]
  band_powers_per_channel: Record<string, Record<string, number>>;
  frontal_specificity:     number;
  suggestibility?:         "high" | "low";
}

// ── Constantes ───────────────────────────────────────────────────────────────

const RECONNECT_DELAY_MS  = 3000;   // 3 s entre intentos de reconexión
const REQUEST_TIMEOUT_MS  = 1000;   // 1 s máximo por request (< hop de 256 ms × 4)
const MAX_RECONNECT_TRIES = 10;

// ── Clase cliente ────────────────────────────────────────────────────────────

export class MLServiceClient {
  private ws: WebSocket | null = null;
  private connected              = false;
  private reconnectAttempts      = 0;
  private readonly url: string;

  // Promise resolvers pendientes: { requestId → {resolve, reject, timer} }
  private pendingRequests = new Map<
    string,
    { resolve: (v: MLServiceResult) => void; reject: (e: Error) => void; timer: NodeJS.Timeout }
  >();

  private requestCounter = 0;

  constructor(url: string = "ws://localhost:8001/ws") {
    this.url = url;
  }

  // ── Conexión ───────────────────────────────────────────────────────────────

  connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.ws = new WebSocket(this.url);

    this.ws.on("open", () => {
      this.connected        = true;
      this.reconnectAttempts = 0;
      console.log("[ml-client] Conectado al servicio Python de análisis EEG");
    });

    this.ws.on("message", (data: Buffer) => {
      try {
        const msg = JSON.parse(data.toString()) as Record<string, unknown>;
        this._handleMessage(msg);
      } catch {
        // Ignorar mensajes mal formados del servicio
      }
    });

    this.ws.on("close", () => {
      this.connected = false;
      this._rejectAllPending(new Error("ML service desconectado"));
      this._scheduleReconnect();
    });

    this.ws.on("error", (err: Error) => {
      // El evento "close" se dispara después de "error",
      // así que la reconexión se maneja ahí.
      console.warn(`[ml-client] Error WS: ${err.message}`);
    });
  }

  disconnect(): void {
    this.reconnectAttempts = MAX_RECONNECT_TRIES; // evitar reconexión
    this.ws?.close();
  }

  get isConnected(): boolean {
    return this.connected;
  }

  // ── Envío de ventana EEG ───────────────────────────────────────────────────

  /**
   * Envía una ventana EEG de 2 s al servicio Python y retorna el resultado.
   *
   * Si el servicio no está disponible, retorna null para que el servidor
   * Node.js pueda continuar enviando el FeedbackPayload sin el módulo ML.
   * Esto es intencional: el módulo de conectividad es un add-on; el pipeline
   * principal de neurofeedback TBR + z-score sigue funcionando sin él.
   *
   * Timeout de 1 s: si el servicio tarda más de 1 s en responder, algo
   * está mal (bucle en Python, carga del servidor). Se rechaza la promesa
   * para no bloquear el próximo hop.
   */
  async processWindow(payload: ProcessWindowPayload): Promise<MLServiceResult | null> {
    if (!this.connected || !this.ws) {
      return null;
    }

    const requestId = `req_${++this.requestCounter}`;

    return new Promise<MLServiceResult | null>((resolve) => {
      const timer = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        console.warn(`[ml-client] Timeout en request ${requestId}`);
        resolve(null);
      }, REQUEST_TIMEOUT_MS);

      this.pendingRequests.set(requestId, {
        resolve: (v) => { clearTimeout(timer); resolve(v); },
        reject:  (e) => { clearTimeout(timer); console.warn(`[ml-client] ${e.message}`); resolve(null); },
        timer,
      });

      this.ws!.send(JSON.stringify({
        type:      "process_window",
        requestId, // el servidor Python devuelve este id para correlacionar
        data:      payload,
      }));
    });
  }

  // ── Manejo de mensajes ─────────────────────────────────────────────────────

  private _handleMessage(msg: Record<string, unknown>): void {
    if (msg.type === "connectivity_result") {
      // Buscar por requestId (si el servidor Python lo soporta) o usar FIFO
      const requestId = msg.requestId as string | undefined;
      const pending   = requestId
        ? this.pendingRequests.get(requestId)
        : this.pendingRequests.values().next().value;

      if (pending) {
        if (requestId) this.pendingRequests.delete(requestId);
        else            this.pendingRequests.delete(this.pendingRequests.keys().next().value!);
        pending.resolve(msg as unknown as MLServiceResult);
      }
    } else if (msg.type === "error") {
      // Rechazar la solicitud pendiente más antigua
      const first = this.pendingRequests.entries().next().value;
      if (first) {
        this.pendingRequests.delete(first[0]);
        first[1].reject(new Error(String(msg.message ?? "error del servicio ML")));
      }
    }
  }

  private _rejectAllPending(err: Error): void {
    for (const [, pending] of this.pendingRequests) {
      clearTimeout(pending.timer);
      pending.reject(err);
    }
    this.pendingRequests.clear();
  }

  private _scheduleReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_TRIES) {
      console.warn("[ml-client] Máximo de reconexiones alcanzado. El módulo ML estará inactivo.");
      return;
    }
    this.reconnectAttempts++;
    console.log(`[ml-client] Reconectando en ${RECONNECT_DELAY_MS / 1000}s (intento ${this.reconnectAttempts})...`);
    setTimeout(() => this.connect(), RECONNECT_DELAY_MS);
  }
}
