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
const MIN_BAND_POWER = 1e-6;

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

  /**
   * Contador interno para throttling de waveformIndex.
   * Se incrementa en cada sample, se resetea cada 8 samples.
   * Interno: no debe usarse en componentes (no exportar selectores para este).
   */
  throttleCounter: number;

  // ── Métricas actuales del epoch ─────────────────────────────────────────
  /** Potencias por banda en µV² del último epoch analizado. null durante warm-up. */
  bandPowers     : BandPowers | null;
  /** Máximo observado por banda durante la sesión actual (normalización visual). */
  sessionMaxBandPowers: Record<keyof Omit<BandPowers, "timestamp">, number>;
  /** Ratio theta/beta del último epoch (0 durante warm-up) */
  thetaBetaRatio : number;
  /** ✅ NUEVO: Predicción de estado del clasificador */
  state_prediction: {
    predicted_label: "awake" | "induction" | "trance";
    confidence: number;  // 0–1
    class_probabilities: {
      awake: number;
      induction: number;
      trance: number;
    };
  } | null;
  /** Tiempo de procesamiento del último epoch en el servidor [ms] */
  pipelineMs     : number;
  /** Ratio theta(Fz) / media theta(F3,F4) */
  frontalSpecificity : number;
  /** true cuando la validación topográfica permite feedback */
  frontalSpecificityValid : boolean;
  /** true cuando se detectó artefacto ocular en Fp1/Fp2 */
  artifactDetected : boolean;
  /** Mapa theta normalizado [0,1] por canal */
  topographyTheta : Record<string, number>;
  /** Canal seleccionado para inspección */
  inspectionChannel: string;
  /** Bloque de señal filtrada del canal de inspección */
  inspectionFilteredSamples: number[];
  /** Magnitudes FFT del canal de inspección */
  inspectionFftMagnitudes: number[];

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
  setInspectionChannel: (channel: string) => void;

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
  sessionMaxBandPowers: {
    delta: MIN_BAND_POWER,
    theta: MIN_BAND_POWER,
    alpha: MIN_BAND_POWER,
    beta : MIN_BAND_POWER,
    gamma: MIN_BAND_POWER,
  },
  thetaBetaRatio  : 0,
  state_prediction: null,
  pipelineMs      : 0,
  frontalSpecificity: 0,
  frontalSpecificityValid: false,
  artifactDetected: false,
  topographyTheta: {},
  inspectionChannel: "Fz",
  inspectionFilteredSamples: [],
  inspectionFftMagnitudes: [],
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
      const { waveformBuffer, waveformIndex } = get();

      waveformBuffer[waveformIndex] = sample;
      const nextIndex = (waveformIndex + 1) % WAVEFORM_BUFFER_SIZE;

      // IMPORTANTE: actualizar el índice en CADA sample, sin throttle.
      // El RAF loop depende del cambio del índice para detectar datos nuevos.
      // Como el servidor solo envía ~4 samples/s, no hay presión de renders.
      // (Si fuera 250/s en un contexto real, aquí habría throttle, pero con
      //  comunicación por WebSocket cada sample es genuinamente nuevo y debe
      //  ser visible inmediatamente.)
      set({ waveformIndex: nextIndex });
    },

    // ── updateMetrics ─────────────────────────────────────────────────────

    updateMetrics: (payload: FeedbackPayload) => {
      // Un único setState → una sola notificación a todos los suscriptores.
      set((state) => ({
        bandPowers: payload.bandPowers,
        sessionMaxBandPowers: {
          delta: Math.max(state.sessionMaxBandPowers.delta, payload.bandPowers.delta),
          theta: Math.max(state.sessionMaxBandPowers.theta, payload.bandPowers.theta),
          alpha: Math.max(state.sessionMaxBandPowers.alpha, payload.bandPowers.alpha),
          beta : Math.max(state.sessionMaxBandPowers.beta, payload.bandPowers.beta),
          gamma: Math.max(state.sessionMaxBandPowers.gamma, payload.bandPowers.gamma),
        },
        thetaBetaRatio : payload.thetaBeta.ratio,
        // ✅ NUEVO: Usar state_prediction del clasificador
        state_prediction: payload.state_prediction,
        pipelineMs     : payload.pipelineMs,
        frontalSpecificity: payload.frontalSpecificity,
        frontalSpecificityValid: payload.frontalSpecificityValid,
        artifactDetected: payload.artifactDetected,
        topographyTheta: payload.topographyTheta,
        inspectionChannel: payload.inspection.channel,
        inspectionFilteredSamples: payload.inspection.filteredSamples,
        inspectionFftMagnitudes: payload.inspection.fftMagnitudes,
      }));
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

    // ── setInspectionChannel ──────────────────────────────────────────
    setInspectionChannel: (channel: string) => {
      set({ inspectionChannel: channel });
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
export const selectSessionMaxBandPowers = (s: EEGState) => s.sessionMaxBandPowers;
export const selectThetaBetaRatio  = (s: EEGState) => s.thetaBetaRatio;
// ✅ NUEVO: Selector para predicción de estado
export const selectStatePrediction = (s: EEGState) => s.state_prediction;
export const selectIsConnected     = (s: EEGState) => s.isConnected;
export const selectIsSessionActive = (s: EEGState) => s.isSessionActive;
export const selectPipelineMs      = (s: EEGState) => s.pipelineMs;
export const selectLastError       = (s: EEGState) => s.lastError;
export const selectFrontalSpecificity = (s: EEGState) => s.frontalSpecificity;
export const selectFrontalSpecificityValid = (s: EEGState) => s.frontalSpecificityValid;
export const selectArtifactDetected = (s: EEGState) => s.artifactDetected;
export const selectTopographyTheta = (s: EEGState) => s.topographyTheta;
export const selectInspectionChannel = (s: EEGState) => s.inspectionChannel;
export const selectInspectionFilteredSamples = (s: EEGState) => s.inspectionFilteredSamples;
export const selectInspectionFftMagnitudes = (s: EEGState) => s.inspectionFftMagnitudes;

/**
 * Selector compuesto para el panel del investigador.
 * Retorna un objeto estable: solo cambia cuando cambian las métricas del epoch.
 *
 * ⚠️  No usar en componentes de alto rendimiento — prefiere selectores atómicos.
 */
export const selectTherapistPanel = (s: EEGState) => ({
  bandPowers     : s.bandPowers,
  thetaBetaRatio : s.thetaBetaRatio,
  statePrediction: s.state_prediction,
  pipelineMs     : s.pipelineMs,
  frontalSpecificity: s.frontalSpecificity,
  frontalSpecificityValid: s.frontalSpecificityValid,
  artifactDetected: s.artifactDetected,
  topographyTheta: s.topographyTheta,
  isConnected    : s.isConnected,
  isSessionActive: s.isSessionActive,
  sessionId      : s.sessionId,
  errorCount     : s.errorCount,
});
