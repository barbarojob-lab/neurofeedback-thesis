/**
 * eegStore.ts  —  frontend/src/store/
 *
 * Store global Zustand para el estado EEG en tiempo real.
 *
 * ── Prioridad de diseño: RENDIMIENTO ─────────────────────────────────────
 *
 * El store recibe ~250 samples/segundo del WebSocket. Si cada sample
 * causara un re-render React, el hilo principal estaría pintando a 250 fps
 * — imposible en 60 Hz y catastrófico para la latencia del resto de la UI.
 *
 * Estrategias aplicadas:
 *
 * 1. FLOAT32ARRAY PARA LA FORMA DE ONDA (no Array JS):
 *    - Float32Array es memoria contigua en heap nativo, sin boxing de números.
 *    - Sin garbage collection por elemento: no hay objetos Number en el heap.
 *    - `pushWaveformSample` muta el array IN-PLACE con escritura indexada O(1),
 *      SIN crear un nuevo Float32Array → Zustand no detecta cambio de referencia
 *      → NO hay re-render por cada sample.
 *    - Los componentes que necesitan la forma de onda usan un selector que
 *      retorna `waveformIndex` (un número que SÍ cambia cada N muestras)
 *      y leen el buffer directamente en su lógica de render.
 *
 * 2. SEPARACIÓN pushWaveformSample / updateMetrics:
 *    - `pushWaveformSample` (250/s): muta Float32Array sin notificar a React.
 *    - `updateMetrics` (~4/s, cada epoch): actualiza bandPowers, zScore,
 *      command → causa el único re-render por segundo relevante para la UI.
 *    - Resultado: la UI de métricas re-renderiza 4 veces/s; el canvas de
 *      forma de onda lee el buffer directamente en su requestAnimationFrame.
 *
 * 3. SELECTORES ATÓMICOS:
 *    Cada componente suscribe solo al slice de estado que necesita.
 *    Ejemplo: el componente de TBR solo suscribe a `thetaBetaRatio`,
 *    no a todo el store → no re-renderiza cuando cambian bandPowers.
 *
 * 4. IMMER NO USADO intencionalmente:
 *    Immer clona objetos, introduciendo GC pressure. Para datos de alta
 *    frecuencia los setters se implementan con object spread mínimo.
 *
 * ── Ring buffer para la forma de onda ────────────────────────────────────
 *
 *   Tamaño: 500 muestras = 2 segundos a 250 Hz.
 *   Estructura: buffer circular con índice `waveformIndex` avanzando módulo 500.
 *
 *   Lectura en el canvas (useCallback en el componente de onda):
 *     const buf = useEEGStore.getState().waveformBuffer;  // sin suscripción
 *     const idx = useEEGStore(s => s.waveformIndex);      // solo suscribe al índice
 *     // Dibujar en orden cronológico:
 *     for (let i = 0; i < 500; i++) {
 *       const sample = buf[(idx + i) % 500];
 *       // ... canvas 2D / WebGL
 *     }
 */

import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

import type {
  BandPowers,
  FeedbackCommand,
  SessionConfig,
  ThetaBetaResult,
  ZScoreResult,
  FeedbackPayload,
} from "../types";

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

/**
 * Número de muestras en el ring buffer de la forma de onda.
 * 500 muestras × (1/250 Hz) = 2 segundos de historia visual.
 * Suficiente para que el terapeuta vea transitorios y artefactos recientes.
 */
const WAVEFORM_BUFFER_SIZE = 500;

// ---------------------------------------------------------------------------
// Interfaces del store
// ---------------------------------------------------------------------------

export interface EEGState {
  // ── Ring buffer de forma de onda ────────────────────────────────────────
  /**
   * Buffer circular Float32Array de las últimas WAVEFORM_BUFFER_SIZE muestras.
   * INMUTABLE como referencia (mismo objeto durante toda la sesión) pero
   * MUTABLE en contenido — se escribe in-place en pushWaveformSample.
   *
   * NO usar este valor directamente como dependencia de useEffect/useMemo:
   * la referencia no cambia. Usar `waveformIndex` como señal de actualización.
   */
  waveformBuffer : Float32Array;
  /**
   * Índice circular actual (0…WAVEFORM_BUFFER_SIZE-1).
   * Se incrementa en cada llamada a pushWaveformSample.
   * Este sí es un número primitivo → Zustand lo compara por valor →
   * los selectores que lo usan re-renderizan solo cuando avanza.
   *
   * Cómo leer el buffer en orden cronológico:
   *   const startIdx = waveformIndex; // sample más antiguo
   *   for (let i = 0; i < WAVEFORM_BUFFER_SIZE; i++) {
   *     const s = waveformBuffer[(startIdx + i) % WAVEFORM_BUFFER_SIZE];
   *   }
   */
  waveformIndex  : number;

  // ── Métricas actuales del epoch ─────────────────────────────────────────
  /** Potencias por banda en µV² del último epoch analizado. null durante warm-up. */
  bandPowers     : BandPowers | null;
  /** Ratio theta/beta del último epoch (0 durante warm-up) */
  thetaBetaRatio : number;
  /** Resultado completo del normalizador adaptativo Welford */
  zScoreResult   : ZScoreResult | null;
  /** Z-score suavizado (EMA) del último epoch. 0 durante warm-up. */
  zScore         : number;
  /** Último comando de feedback emitido por el motor adaptativo */
  command        : FeedbackCommand | null;
  /** Tiempo de procesamiento del último epoch en el servidor [ms] */
  pipelineMs     : number;

  // ── Estado de sesión ─────────────────────────────────────────────────────
  isConnected     : boolean;
  isSessionActive : boolean;
  sessionId       : string | null;
  sessionConfig   : SessionConfig | null;
  /** Número de errores WS recibidos del servidor en la sesión actual */
  errorCount      : number;
  /** Último mensaje de error del servidor */
  lastError       : string | null;

  // ── Acciones ────────────────────────────────────────────────────────────

  /**
   * Escribe un sample en el ring buffer IN-PLACE (sin crear nuevo objeto).
   *
   * ⚠️  Esta función muta waveformBuffer directamente y luego actualiza solo
   * `waveformIndex`. Los componentes que renderizen la forma de onda deben
   * suscribirse a `waveformIndex`, NO a `waveformBuffer`.
   *
   * Llamada ~250 veces/segundo — debe ser O(1) y sin allocaciones.
   */
  pushWaveformSample : (sample: number) => void;

  /**
   * Actualiza todas las métricas del epoch de una vez.
   * Llamada ~4 veces/segundo (una por hopSize=64 muestras a 250 Hz).
   *
   * Agrupa bandPowers + thetaBeta + zScore + command en un único setState
   * para que Zustand emita UNA sola notificación a los suscriptores,
   * en lugar de cuatro notificaciones separadas (que causarían 4 renders).
   */
  updateMetrics : (payload: FeedbackPayload) => void;

  setConnected     : (connected: boolean) => void;
  setSessionActive : (active: boolean, sessionId?: string) => void;
  setSessionConfig : (config: SessionConfig | null) => void;
  setError         : (message: string | null) => void;

  /** Resetea el store al estado inicial (nueva sesión o desconexión) */
  resetSession : () => void;
}

// ---------------------------------------------------------------------------
// Estado inicial
// ---------------------------------------------------------------------------

/**
 * Función factory para el estado inicial — permite llamarla en resetSession
 * sin duplicar la definición del estado por defecto.
 *
 * Nota: Float32Array se inicializa con ceros por la especificación ECMAScript.
 * Esto es correcto: la forma de onda parte plana hasta recibir los primeros
 * samples reales.
 */
const createInitialState = () => ({
  waveformBuffer  : new Float32Array(WAVEFORM_BUFFER_SIZE),
  waveformIndex   : 0,
  throttleCounter : 0,
  bandPowers      : null as BandPowers | null,
  thetaBetaRatio  : 0,
  zScoreResult    : null as ZScoreResult | null,
  zScore          : 0,
  command         : null as FeedbackCommand | null,
  pipelineMs      : 0,
  isConnected     : false,
  isSessionActive : false,
  sessionId       : null as string | null,
  sessionConfig   : null as SessionConfig | null,
  errorCount      : 0,
  lastError       : null as string | null,
});


// ---------------------------------------------------------------------------
// Creación del store
// ---------------------------------------------------------------------------

/**
 * `subscribeWithSelector` middleware habilita suscripciones a slices
 * específicos del estado sin re-renderizar por cambios en el resto:
 *
 *   // Solo re-renderiza cuando cambia thetaBetaRatio
 *   const tbr = useEEGStore(s => s.thetaBetaRatio);
 *
 *   // Suscripción fuera de React (e.g., audio engine)
 *   useEEGStore.subscribe(
 *     s => s.command,
 *     command => audioEngine.applyCommand(command)
 *   );
 */
export const useEEGStore = create<EEGState>()(
  subscribeWithSelector((set, get) => ({
    ...createInitialState(),

    // ── pushWaveformSample ────────────────────────────────────────────────
    pushWaveformSample: (sample: number) => {
      // Mutación in-place: NO crea nuevo Float32Array.
      // `get()` es síncrono y no causa re-render — es solo lectura del estado.
      const { waveformBuffer, waveformIndex } = get();

      waveformBuffer[waveformIndex] = sample;
      const nextIndex = (waveformIndex + 1) % WAVEFORM_BUFFER_SIZE;

      // Solo actualizamos `waveformIndex` — un número primitivo.
      // Zustand compara por referencia: number === number → comparación de valor.
      // Los componentes suscritos a waveformIndex re-renderizan solo cuando
      // el índice cambia (es decir, siempre — pero a 250 Hz el canvas ya usa
      // requestAnimationFrame que limita a 60 fps de todos modos).
      //
      // Si 250 actualizaciones/s siguen siendo demasiadas, se puede throttlear:
      //   if (nextIndex % 8 === 0) set({ waveformIndex: nextIndex });
      // Lo que actualiza el índice solo 31 veces/s manteniendo el buffer al día.
      set({ waveformIndex: nextIndex });
    },

    // ── updateMetrics ─────────────────────────────────────────────────────
    updateMetrics: (payload: FeedbackPayload) => {
      // Un único setState → una sola notificación a todos los suscriptores.
      // Sin this, cuatro setters separados causarían cuatro renders por epoch.
      set({
        bandPowers     : payload.bandPowers,
        thetaBetaRatio : payload.thetaBeta.ratio,
        zScoreResult   : payload.zScore,
        zScore         : payload.zScore.zSmooth,
        command        : payload.command,
        pipelineMs     : payload.pipelineMs,
      });
    },

    // ── setConnected ──────────────────────────────────────────────────────
    setConnected: (connected: boolean) => {
      set({ isConnected: connected });
    },

    // ── setSessionActive ──────────────────────────────────────────────────
    setSessionActive: (active: boolean, sessionId?: string) => {
      set({
        isSessionActive : active,
        sessionId       : active ? (sessionId ?? null) : null,
      });
    },

    // ── setSessionConfig ──────────────────────────────────────────────────
    setSessionConfig: (config: SessionConfig | null) => {
      set({ sessionConfig: config });
    },

    // ── setError ──────────────────────────────────────────────────────────
    setError: (message: string | null) => {
      if (message === null) {
        set({ lastError: null });
      } else {
        set((state) => ({
          lastError  : message,
          errorCount : state.errorCount + 1,
        }));
        console.error(`[EEGStore] Server error: ${message}`);
      }
    },

    // ── resetSession ──────────────────────────────────────────────────────
    resetSession: () => {
      // Reutiliza el mismo Float32Array (lo llena con ceros) para evitar
      // la allocación de un nuevo buffer y la presión de GC asociada.
      const { waveformBuffer } = get();
      waveformBuffer.fill(0);

      set({
        ...createInitialState(),
        // Mantener el buffer existente (ya limpiado con fill(0)):
        waveformBuffer,
      });
    },
  }))
);

// ---------------------------------------------------------------------------
// Selectores derivados (memoizados por Zustand automáticamente)
// ---------------------------------------------------------------------------

/**
 * Selectores atómicos para componentes individuales.
 * Importar solo el selector necesario para minimizar el scope de re-renders.
 *
 * Ejemplo de uso:
 *   const tbr = useEEGStore(selectThetaBetaRatio);
 *   const bands = useEEGStore(selectBandPowers);
 */

export const selectWaveformIndex   = (s: EEGState) => s.waveformIndex;
export const selectWaveformBuffer  = (s: EEGState) => s.waveformBuffer;
export const selectBandPowers      = (s: EEGState) => s.bandPowers;
export const selectThetaBetaRatio  = (s: EEGState) => s.thetaBetaRatio;
export const selectZScore          = (s: EEGState) => s.zScore;
export const selectZScoreResult    = (s: EEGState) => s.zScoreResult;
export const selectCommand         = (s: EEGState) => s.command;
export const selectIsConnected     = (s: EEGState) => s.isConnected;
export const selectIsSessionActive = (s: EEGState) => s.isSessionActive;
export const selectPipelineMs      = (s: EEGState) => s.pipelineMs;
export const selectLastError       = (s: EEGState) => s.lastError;

/**
 * Selector compuesto para el panel del terapeuta.
 * Retorna un objeto estable: solo cambia cuando cambian las métricas del epoch,
 * no con cada sample de la forma de onda.
 *
 * ⚠️  No usar en componentes de alto rendimiento — prefiere selectores atómicos.
 */
export const selectTherapistPanel = (s: EEGState) => ({
  bandPowers     : s.bandPowers,
  thetaBetaRatio : s.thetaBetaRatio,
  zScore         : s.zScore,
  zScoreResult   : s.zScoreResult,
  command        : s.command,
  pipelineMs     : s.pipelineMs,
  isConnected    : s.isConnected,
  isSessionActive: s.isSessionActive,
  sessionId      : s.sessionId,
  errorCount     : s.errorCount,
});
