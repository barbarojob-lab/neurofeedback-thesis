/**
 * types.ts  —  frontend/src/
 *
 * Contratos de tipo compartidos entre frontend y backend.
 *
 * ── Nota sobre la duplicación ────────────────────────────────────────────
 * En producción estas interfaces se publicarían como un paquete interno
 * (e.g. @eeg-nf/shared-types) consumido por ambos workspace packages, de
 * modo que un cambio en el backend rompe el build del frontend en CI antes
 * de llegar a producción. Por ahora se copian manualmente; cualquier cambio
 * en el backend debe reflejarse aquí sincrónicamente.
 *
 * Estrategia de validación en runtime:
 * Los mensajes WS llegan como `unknown`. En lugar de castear a ciegas, el
 * hook useEEGSocket usa type guards sobre el campo `type` del mensaje antes
 * de despachar al store. Esto detecta en runtime cualquier drift entre
 * la versión del servidor y del cliente.
 */

// ---------------------------------------------------------------------------
// DSP — BandPowers (espejo de backend/src/dsp/band-power.ts)
// ---------------------------------------------------------------------------

/** Potencias espectrales por banda EEG en µV² */
export interface BandPowers {
  /** Delta 1–4 Hz: somnolencia / disociación en trance profundo */
  delta: number;
  /** Theta 4–8 Hz: ← señal principal de trance/relajación en neurofeedback */
  theta: number;
  /** Alpha 8–12 Hz: relajación alerta, máximo con ojos cerrados */
  alpha: number;
  /** Beta 12–30 Hz: activación cognitiva, arousal cortical */
  beta: number;
  /**
   * Gamma 30–45 Hz: integración sensorial de alto nivel.
   * ⚠️  Solo uso offline — contaminado por EMG facial en tiempo real.
   */
  gamma: number;
  /** Timestamp de la época analizada (ms Unix) */
  timestamp: number;
}

// ---------------------------------------------------------------------------
// DSP — ThetaBetaResult (espejo de backend/src/dsp/band-power.ts)
// ---------------------------------------------------------------------------

/** Estados cognitivos derivados del ratio theta/beta */
export type TBRState = "hyperactive" | "alert" | "relaxed" | "drowsy" | "trance";

/** Resultado del cálculo del ratio theta/beta con clasificación de estado */
export interface ThetaBetaResult {
  /** TBR = P_theta / P_beta (adimensional) */
  ratio: number;
  /** Potencia theta [µV²] */
  thetaPower: number;
  /** Potencia beta [µV²] */
  betaPower: number;
  /** Clasificación clínica del estado cognitivo */
  state: TBRState;
  /** Descripción legible para el panel del terapeuta */
  stateDescription: string;
}

// ---------------------------------------------------------------------------
// Adaptive — ZScoreResult (espejo de backend/src/adaptive/running-zscore.ts)
// ---------------------------------------------------------------------------

/** Resultado del normalizador adaptativo Welford */
export interface ZScoreResult {
  /** Z-score instantáneo sin suavizar */
  zRaw: number;
  /**
   * Z-score suavizado con EMA α=0.1.
   * Este es el valor que debe usar el motor de feedback y la UI.
   * τ ≈ 2.5 s → transiciones suaves sin lag perceptible.
   */
  zSmooth: number;
  /** Media actual de la ventana deslizante de 3 min */
  mean: number;
  /** Desviación estándar actual */
  std: number;
  /** Número de épocas en la ventana activa */
  n: number;
  /**
   * true cuando n ≥ 30: z-score estadísticamente confiable.
   * La UI debe mostrar un indicador "calibrando..." mientras isReady=false.
   */
  isReady: boolean;
}

// ---------------------------------------------------------------------------
// Adaptive — FeedbackCommand (espejo de backend/src/adaptive/feedback-engine.ts)
// ---------------------------------------------------------------------------

/**
 * Acción de feedback generada por el motor adaptativo.
 *
 *   decrease_theta : Beta dominante → guiar hacia relajación (estímulo "alerta")
 *   neutral        : Cerca del baseline → mantener estado (sin cambio)
 *   increase_theta : Theta creciendo → refuerzo positivo (estímulo relajante)
 *   sustain_trance : Trance profundo sostenido → refuerzo máximo fijo
 */
export type FeedbackAction =
  | "decrease_theta"
  | "neutral"
  | "increase_theta"
  | "sustain_trance";

/** Zona del z-score de la que proviene el comando */
export type FeedbackZone =
  | "hyperactive"
  | "below_baseline"
  | "entering_trance"
  | "deep_trance";

/** Comando de feedback listo para consumir por la capa de presentación */
export interface FeedbackCommand {
  /** Acción a ejecutar en la UI */
  action: FeedbackAction;
  /**
   * Intensidad del estímulo en [0.0, 1.0].
   * La UI mapea esto a: volumen de audio, brillo visual, velocidad de animación.
   */
  intensity: number;
  /** Z-score suavizado que originó el comando */
  zScore: number;
  /** true si se detectó un pico theta significativo (TBR + estado convergentes) */
  thetaPeak: boolean;
  /** Timestamp de generación (ms Unix) */
  timestamp: number;
  /** Metadatos para el panel del terapeuta */
  meta: {
    tbrRatio : number;
    tbrState : TBRState;
    zone     : FeedbackZone;
  };
}

// ---------------------------------------------------------------------------
// Sesión
// ---------------------------------------------------------------------------

/** Configuración enviada desde la UI al iniciar una sesión */
export interface SessionConfig {
  /**
   * Umbral de z-score para refuerzo positivo.
   * Default 0: cualquier valor sobre el baseline del paciente es reforzado.
   */
  thetaThreshold?: number;
  /**
   * TBR mínimo para detectar pico theta.
   * Default 2.5 — confirmación por dos métricas independientes.
   */
  thetaPeakTbrThreshold?: number;
  /**
   * Factor k de la sigmoide de mapeo z→intensidad.
   * Default 2.0 — balance responsividad/estabilidad para EEG.
   */
  sigmoidK?: number;
  /** Umbral mínimo de especificidad frontal theta(Fz)/media(theta(F3),theta(F4)) */
  frontalSpecificityThreshold?: number;
  /** Perfil de sugestionabilidad del sujeto para seleccionar modelo ML */
  suggestibility?: "high" | "low";
}

/** Medida subjetiva de estado reportada por el paciente o terapeuta */
export interface SubjectiveMeasure {
  sessionId  : string;
  timestamp  : number;
  /** Etiqueta del instrumento (e.g. "relax_self_report", "depth_of_trance") */
  label      : string;
  /** Valor en escala Likert 0–10 */
  value      : number;
  notes?     : string;
}

// ---------------------------------------------------------------------------
// ML — ClassifierPrediction (espejo de backend/src/ml-client.ts)
// ---------------------------------------------------------------------------

/**
 * Predicción del clasificador ML para el estado del paciente.
 * Basado en ensemble SVM + Random Forest.
 */
export interface ClassifierPrediction {
  /** Estado predicho: 0=awake, 1=induction, 2=trance */
  predicted_class: number;
  /** Etiqueta legible del estado */
  predicted_label: "awake" | "induction" | "trance";
  /** Confianza de la predicción [0, 1] */
  confidence: number;
  /** Probabilidades por clase [normalizado a suma ≈ 1] */
  class_probabilities: {
    awake: number;
    induction: number;
    trance: number;
  };
  /** true cuando confidence >= 0.50 */
  is_confident: boolean;
  /** Método usado: 'ml_ensemble' o 'heuristic_fallback' */
  method: "ml_ensemble" | "heuristic_fallback";
}

/** Conectividad y features del ML service */
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

export interface MLServiceResult {
  coherence_matrix: number[][];
  plv_matrix: number[][];
  channel_labels: string[];
  connectivity_features: ConnectivityFeatures;
  classifier_prediction: ClassifierPrediction;
  feature_vector: number[];
  processing_ms: number;
}

// ---------------------------------------------------------------------------
// Protocolo WebSocket — Mensajes entrantes (server → client)
// ---------------------------------------------------------------------------

/** Payload principal enviado cada ~256 ms (cada hopSize muestras) */
export interface FeedbackPayload {
  type            : "eeg_data";
  timestamp       : number;
  /** Último sample EEG filtrado [µV] — compatibilidad hacia atrás */
  filteredSample  : number;
  /**
   * Todos los samples filtrados del hop actual [µV].
   * Longitud = hopSize (64 a 250 Hz). Usar esto para el osciloscopio:
   * permite dibujar a 250 sps reales en lugar de los 4 muestras/s del
   * filteredSample individual.
   */
  filteredSamples : number[];
  frontalSpecificity      : number;
  frontalSpecificityValid : boolean;
  artifactDetected        : boolean;
  topographyTheta         : Record<string, number>;
  inspection: {
    channel: string;
    filteredSamples: number[];
    fftMagnitudes: number[];
    bandPowers: BandPowers;
  };
  bandPowers      : BandPowers;
  thetaBeta       : ThetaBetaResult;
  // ✅ NUEVO: Predicción de estado del clasificador
  state_prediction: ClassifierPrediction | null;
  // ✅ NUEVO: Resultados de conectividad y ML
  connectivity    : MLServiceResult | null;
  /** Tiempo de procesamiento del epoch en el servidor [ms] — debe ser < 5 ms */
  pipelineMs      : number;
}

export interface DatasetMetadata {
  id: string;
  sourcePath: string;
  format: "edf" | "csv";
  channels: string[];
  sampleRate: number;
  durationSec: number;
  totalSamples: number;
}

export interface ServerHelloMessage {
  type       : "server_hello";
  version    : string;
  sampleRate : number;
  windowSize : number;
  hopSize    : number;
  dataset?   : DatasetMetadata | null;
  session    : { id: string; tranceMode: boolean } | null;
}

export interface SessionStartedMessage {
  type      : "session_started";
  sessionId : string;
}

export interface SessionStoppedMessage {
  type: "session_stopped";
}

export interface PongMessage {
  type      : "pong";
  timestamp : number;
}

export interface SubjectiveSavedMessage {
  type      : "subjective_saved";
  sessionId : string;
}

export interface TranceModSetMessage {
  type    : "trance_mode_set";
  enabled : boolean;
}

export interface InspectionChannelSetMessage {
  type    : "inspection_channel_set";
  channel : string;
}

export interface DatasetLoadedMessage {
  type    : "dataset_loaded";
  dataset : DatasetMetadata;
}

export interface ServerErrorMessage {
  type    : "error";
  message : string;
}

/** Unión discriminada de todos los mensajes server → client */
export type ServerMessage =
  | FeedbackPayload
  | ServerHelloMessage
  | SessionStartedMessage
  | SessionStoppedMessage
  | PongMessage
  | SubjectiveSavedMessage
  | TranceModSetMessage
  | InspectionChannelSetMessage
  | DatasetLoadedMessage
  | ServerErrorMessage;

// ---------------------------------------------------------------------------
// Protocolo WebSocket — Mensajes salientes (client → server)
// ---------------------------------------------------------------------------

export type WSMessage =
  | { type: "start_session";     payload: SessionConfig       }
  | { type: "stop_session"                                    }
  | { type: "set_trance_mode";   payload: { enabled: boolean }}
  | { type: "set_inspection_channel"; payload: { channel: string }}
  | { type: "load_dataset"; payload: { path: string }}
  | { type: "ping"                                            }
  | { type: "submit_subjective"; payload: SubjectiveMeasure   };

// ---------------------------------------------------------------------------
// Type Guards — validación en runtime de mensajes WebSocket
// ---------------------------------------------------------------------------

type UnknownObject = Record<string, unknown>;

function asObject(msg: unknown): UnknownObject | null {
  return typeof msg === "object" && msg !== null
    ? (msg as UnknownObject)
    : null;
}

/**
 * Valida que un objeto desconocido es un FeedbackPayload.
 * Usado en el hook useEEGSocket para descartar mensajes mal formados.
 */
export function isFeedbackPayload(msg: unknown): msg is FeedbackPayload {
  const obj = asObject(msg);
  if (!obj) return false;

  return (
    obj.type === "eeg_data" &&
    typeof obj.timestamp === "number" &&
    typeof obj.filteredSample === "number" &&
    typeof obj.pipelineMs === "number" &&
    typeof obj.frontalSpecificity === "number" &&
    typeof obj.frontalSpecificityValid === "boolean" &&
    typeof obj.artifactDetected === "boolean" &&
    asObject(obj.topographyTheta) !== null &&
    asObject(obj.inspection) !== null &&
    asObject(obj.bandPowers) !== null &&
    asObject(obj.thetaBeta) !== null
    // state_prediction puede ser null, no es obligatorio validarlo
  );
}

/**
 * Valida que un objeto es un ServerHelloMessage.
 */
export function isServerHello(msg: unknown): msg is ServerHelloMessage {
  const obj = asObject(msg);
  if (!obj) return false;

  return (
    obj.type === "server_hello" &&
    typeof obj.version === "string"
  );
}

/**
 * Valida que un objeto es un SessionStartedMessage.
 */
export function isSessionStarted(msg: unknown): msg is SessionStartedMessage {
  const obj = asObject(msg);
  if (!obj) return false;

  return (
    obj.type === "session_started" &&
    typeof obj.sessionId === "string"
  );
}

/**
 * Valida que un objeto es un SessionStoppedMessage.
 */
export function isSessionStopped(msg: unknown): msg is SessionStoppedMessage {
  const obj = asObject(msg);
  return obj !== null && obj.type === "session_stopped";
}

/**
 * Valida que un objeto es un PongMessage.
 */
export function isPong(msg: unknown): msg is PongMessage {
  const obj = asObject(msg);
  if (!obj) return false;

  return (
    obj.type === "pong" &&
    typeof obj.timestamp === "number"
  );
}

/**
 * Valida que un objeto es un ServerErrorMessage.
 */
export function isServerError(msg: unknown): msg is ServerErrorMessage {
  const obj = asObject(msg);
  if (!obj) return false;

  return (
    obj.type === "error" &&
    typeof obj.message === "string"
  );
}

/**
 * Valida que un objeto es un TranceModSetMessage.
 */
export function isTranceModeSet(msg: unknown): msg is TranceModSetMessage {
  const obj = asObject(msg);
  if (!obj) return false;

  return (
    obj.type === "trance_mode_set" &&
    typeof obj.enabled === "boolean"
  );
}

export function isInspectionChannelSet(msg: unknown): msg is InspectionChannelSetMessage {
  const obj = asObject(msg);
  if (!obj) return false;

  return (
    obj.type === "inspection_channel_set" &&
    typeof obj.channel === "string"
  );
}

export function isDatasetLoaded(msg: unknown): msg is DatasetLoadedMessage {
  const obj = asObject(msg);
  if (!obj) return false;

  return obj.type === "dataset_loaded" && asObject(obj.dataset) !== null;
}
