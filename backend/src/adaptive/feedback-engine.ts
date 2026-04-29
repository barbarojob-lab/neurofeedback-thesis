/**
 * feedback-engine.ts  —  backend/src/adaptive/
 *
 * Mapea el z-score normalizado del paciente (RunningZScore) y el estado
 * theta/beta (BandPowerExtractor) a comandos concretos de feedback para
 * la interfaz visual/auditiva del sistema de neurofeedback.
 *
 * ── Arquitectura del motor de feedback ──────────────────────────────────
 *
 *   RunningZScore.zSmooth  ──┐
 *                            ├──► FeedbackEngine.computeCommand() ──► FeedbackCommand
 *   ThetaBetaResult ─────────┘
 *
 * ── Diseño de la función de mapeo z → intensidad ────────────────────────
 *
 *   Se usa la función sigmoide en lugar de un mapeo lineal por dos razones:
 *
 *   1. SATURACIÓN SUAVE en extremos:
 *      Un z-score de +3 y uno de +5 deben producir intensidades similares
 *      (ambos son "trance profundo"). El mapeo lineal desbordaría el rango
 *      [0,1] y requeriría clipping explícito. La sigmoide satura suavemente.
 *
 *   2. SENSIBILIDAD ALTA en el centro (z ≈ 0):
 *      El gradiente máximo de σ(x) ocurre en x=0. Esto significa que
 *      pequeñas variaciones del z-score cerca del baseline del paciente
 *      producen cambios de intensidad más pronunciados → feedback más
 *      responsivo justo cuando el paciente está transitando entre estados.
 *      En los extremos (trance profundo o muy activado), la intensidad
 *      ya está cerca de 1.0 o 0.0 y variaciones adicionales no cambian
 *      el feedback perceptiblemente — correcto, porque el paciente ya
 *      recibió la señal.
 *
 *   Sigmoide parametrizada:
 *     sigmoid(x, k) = 1 / (1 + e^{−k·x})
 *     k=2: pendiente moderada, buen balance respuesta/estabilidad para EEG
 *
 * ── Zonas del z-score y su correspondencia clínica ──────────────────────
 *
 *   z < −1.5 : "Muy activado" → acción 'decrease_theta'
 *     Beta dominante, arousal alto. El feedback debe guiar AL PACIENTE
 *     hacia abajo (relajar): feedback de alerta (tono agudo, luz intensa).
 *
 *   −1.5 ≤ z < 0 : "Baseline bajo" → acción 'neutral'
 *     Ligeramente por debajo del baseline propio. Mantener estado actual.
 *     Intensidad fija 0.5 = feedback neutro, sin refuerzo ni penalización.
 *
 *   0 ≤ z < 1.5 : "Entrando en trance" → acción 'increase_theta'
 *     Por encima del baseline: theta aumentando. Reforzar con feedback
 *     positivo proporcional (tono grave, luz suave, música relajante).
 *     Intensidad = sigmoid(z, 2) ∈ [0.5, 0.88]
 *
 *   z ≥ 1.5 : "Trance profundo" → acción 'increase_theta', intensidad máxima
 *     Feedback de refuerzo máximo sostenido. No escalar más porque puede
 *     despertar al paciente si el estímulo es demasiado intenso.
 *     Intensidad fija = 1.0
 */

import type { ThetaBetaResult } from "../dsp/band-power";
import type { ZScoreResult }    from "./running-zscore";

// ---------------------------------------------------------------------------
// Tipos importados / declarados localmente
// (en el proyecto real, estos vendrían de ../types)
// ---------------------------------------------------------------------------

export type FeedbackAction =
  | "decrease_theta" // Arousal alto → guiar hacia relajación
  | "neutral"        // Cerca del baseline → mantener estado
  | "increase_theta" // Theta creciente → reforzar positivamente
  | "sustain_trance" // Trance profundo sostenido → mantener sin excitar

/** Comando de feedback listo para consumir por la capa de presentación */
export interface FeedbackCommand {
  /** Acción principal a ejecutar en la UI */
  action: FeedbackAction;
  /**
   * Intensidad del estímulo en [0.0, 1.0].
   * 0.0 = mínimo perceptible, 1.0 = máximo del rango configurado.
   * La capa de presentación mapea esto a: volumen de audio, brillo visual,
   * frecuencia de parpadeo, etc.
   */
  intensity: number;
  /** Z-score suavizado que originó este comando (para logging/debug) */
  zScore: number;
  /** true si se detectó un pico theta significativo en este epoch */
  thetaPeak: boolean;
  /** Timestamp de generación del comando (ms Unix) */
  timestamp: number;
  /** Metadatos opcionales para el panel del terapeuta */
  meta: {
    tbrRatio: number;
    tbrState: ThetaBetaResult["state"];
    zone: "hyperactive" | "below_baseline" | "entering_trance" | "deep_trance";
  };
}

/** Configuración de sesión (pasa desde el nivel de sesión) */
export interface SessionConfig {
  /**
   * Umbral de z-score para considerar que el paciente está "entrando en trance".
   * Default 0: cualquier valor por encima del baseline personal es reforzado.
   */
  thetaThreshold?: number;
  /**
   * Umbral de ratio theta/beta para detectar un pico theta relevante.
   * Default 2.5 (TBR > 2.5 = relajación profunda confirmada por dos métricas).
   */
  thetaPeakTbrThreshold?: number;
  /**
   * Factor k de la sigmoide. Valores altos → transiciones más abruptas.
   * Default 2.0 (recomendado para neurofeedback; valores > 4 son muy reactivos).
   */
  sigmoidK?: number;
  /** Umbral mínimo de especificidad frontal (theta Fz / media theta F3-F4) */
  frontalSpecificityThreshold?: number;
}

/**
 * Valida y normaliza una SessionConfig.
 * Retorna una copia con valores por defecto y rangos garantizados.
 *
 * @param config Configuración potencialmente incompleta o fuera de rango
 * @returns      Configuración normalizada y validada
 */
export function validateSessionConfig(config: SessionConfig = {}): Required<SessionConfig> {
  let thetaThreshold = config.thetaThreshold ?? 0;
  let thetaPeakTbrThreshold = config.thetaPeakTbrThreshold ?? 2.5;
  let sigmoidK = config.sigmoidK ?? 2.0;
  let frontalSpecificityThreshold = config.frontalSpecificityThreshold ?? 1.5;

  // Validar rangos y sanitizar valores fuera de límites
  if (!Number.isFinite(thetaThreshold)) {
    console.warn("[validateSessionConfig] thetaThreshold no es un número válido, usando default 0");
    thetaThreshold = 0;
  }

  if (!Number.isFinite(thetaPeakTbrThreshold) || thetaPeakTbrThreshold <= 0) {
    console.warn("[validateSessionConfig] thetaPeakTbrThreshold inválido, usando default 2.5");
    thetaPeakTbrThreshold = 2.5;
  }

  if (!Number.isFinite(sigmoidK) || sigmoidK <= 0) {
    console.warn("[validateSessionConfig] sigmoidK debe ser > 0, usando default 2.0");
    sigmoidK = 2.0;
  }

  if (!Number.isFinite(frontalSpecificityThreshold) || frontalSpecificityThreshold <= 0) {
    console.warn("[validateSessionConfig] frontalSpecificityThreshold inválido, usando default 1.5");
    frontalSpecificityThreshold = 1.5;
  }

  // Limitar sigmoidK a rango razonable (0.1 a 10)
  if (sigmoidK > 10) {
    console.warn("[validateSessionConfig] sigmoidK > 10 es muy reactivo, limitando a 10");
    sigmoidK = 10;
  }
  if (sigmoidK < 0.1) {
    console.warn("[validateSessionConfig] sigmoidK < 0.1 es muy inerte, limitando a 0.1");
    sigmoidK = 0.1;
  }

  return { thetaThreshold, thetaPeakTbrThreshold, sigmoidK, frontalSpecificityThreshold };
}

// ---------------------------------------------------------------------------
// Clase principal
// ---------------------------------------------------------------------------

export class FeedbackEngine {
  private readonly thetaThreshold:      number;
  private readonly thetaPeakTbrThresh:  number;
  private readonly sigmoidK:            number;

  // Historial de los últimos N comandos para detectar tendencia sostenida
  private readonly commandHistory: FeedbackAction[] = [];
  private readonly historySize = 8; // ~2 s a 4 epochs/s

  constructor(config: SessionConfig = {}) {
    const validated = validateSessionConfig(config);
    this.thetaThreshold     = validated.thetaThreshold;
    this.thetaPeakTbrThresh = validated.thetaPeakTbrThreshold;
    this.sigmoidK           = validated.sigmoidK;
  }

  // ---------------------------------------------------------------------------
  // Método principal
  // ---------------------------------------------------------------------------

  /**
   * Genera un FeedbackCommand a partir del z-score actual del paciente
   * y del resultado theta/beta del epoch.
   *
   * La lógica de zonas es intencionalmente asimétrica:
   *   - La zona de "decrease" requiere z < −1.5 (umbral estricto): no queremos
   *     penalizar al paciente por estar ligeramente por debajo de su baseline.
   *   - La zona de "increase" comienza en z = 0 (baseline): cualquier mejora
   *     sobre el propio promedio reciente merece refuerzo positivo.
   *   Esta asimetría está respaldada por la teoría del condicionamiento operante
   *   aplicado a neurofeedback (Skinner, 1938; Sterman, 2000): el refuerzo
   *   positivo es más efectivo que el negativo para aprendizaje de habilidades
   *   de autorregulación.
   *
   * @param zScore    Resultado de RunningZScore.push() (usar zSmooth)
   * @param thetaBeta Resultado de BandPowerExtractor.computeThetaBetaRatio()
   * @returns         FeedbackCommand listo para la capa de presentación
   */
  computeCommand(
    zScore: number | ZScoreResult,
    thetaBeta: ThetaBetaResult
  ): FeedbackCommand {
    // Aceptar tanto el z-score numérico directo como el ZScoreResult completo
    const z = typeof zScore === "number" ? zScore : zScore!.zSmooth;

    const thetaPeak = this.detectThetaPeak(thetaBeta, this.thetaPeakTbrThresh);

    let action:    FeedbackAction;
    let intensity: number;
    let zone: FeedbackCommand["meta"]["zone"];

    if (z < -1.5) {
      // ── Zona 1: Muy activado / arousal alto ─────────────────────────
      // Beta muy dominante sobre el baseline del paciente.
      // Feedback: señal de "baja el arousal" (tono agudo, luz neutra/fría).
      // Intensidad: sigmoid(-z, k) → crece cuanto MÁS activado está,
      // así que el estímulo de "alerta" se intensifica proporcionalmente.
      action    = "decrease_theta";
      intensity = this.sigmoid(-z, this.sigmoidK); // -z > 1.5 → sigmoid > 0.75
      zone      = "hyperactive";

    } else if (z < this.thetaThreshold) {
      // ── Zona 2: Ligeramente bajo baseline ───────────────────────────
      // No penalizar: el paciente está en su estado normal o levemente
      // por debajo. Feedback neutro: sin cambio de estímulo.
      action    = "neutral";
      intensity = 0.5; // 0.5 = "sin cambio" en la escala [0,1] de la UI
      zone      = "below_baseline";

    } else if (z < 1.5) {
      // ── Zona 3: Entrando en trance ──────────────────────────────────
      // Theta aumentando sobre el baseline. Refuerzo positivo proporcional.
      // sigmoid(z, k) ∈ [0.5, 0.88] para z ∈ [0, 1.5) con k=2.
      action    = "increase_theta";
      intensity = this.sigmoid(z, this.sigmoidK);
      zone      = "entering_trance";

    } else {
      // ── Zona 4: Trance profundo ──────────────────────────────────────
      // z ≥ 1.5: theta muy por encima del baseline personal.
      // Feedback máximo pero FIJO en 1.0 para no sobreestimular.
      // Un estímulo demasiado intenso durante trance profundo puede
      // interrumpir el estado (efecto de "despertar por refuerzo").
      action    = "sustain_trance";
      intensity = 1.0;
      zone      = "deep_trance";
    }

    // Registrar en historial de tendencia
    this._recordHistory(action);

    return {
      action,
      intensity,
      zScore    : z,
      thetaPeak,
      timestamp : Date.now(),
      meta: {
        tbrRatio : thetaBeta.ratio,
        tbrState : thetaBeta.state,
        zone,
      },
    };
  }

  /**
   * Detecta si el epoch actual contiene un pico theta significativo,
   * definido como la convergencia de DOS criterios independientes:
   *
   *   1. El ratio theta/beta (TBR) supera el umbral configurado.
   *   2. La clasificación de estado de ThetaBetaResult indica
   *      relajación, somnolencia o trance (no alerta/hiperactivación).
   *
   * Requerir DOS criterios reduce los falsos positivos causados por:
   *   - Un pico aislado de theta sin cambio en beta (ruido espectral).
   *   - Un beta muy bajo (sujeto dormido) que infla el TBR sin theta real.
   *
   * @param thetaBeta  Resultado de BandPowerExtractor.computeThetaBetaRatio()
   * @param threshold  TBR mínimo para considerar pico theta
   * @returns          true si ambos criterios se cumplen simultáneamente
   */
  detectThetaPeak(
    thetaBeta: ThetaBetaResult,
    threshold: number = this.thetaPeakTbrThresh
  ): boolean {
    const tbrCriteria   = thetaBeta.ratio >= threshold;
    const stateCriteria = (
      thetaBeta.state === "relaxed" ||
      thetaBeta.state === "drowsy"  ||
      thetaBeta.state === "trance"
    );
    return tbrCriteria && stateCriteria;
  }

  /**
   * Retorna true si los últimos N comandos consecutivos son todos de la
   * misma acción → indica una tendencia sostenida (útil para el terapeuta).
   *
   * @param action Acción a verificar
   * @param streak Número de comandos consecutivos requeridos (default 4)
   */
  isSustainedTrend(action: FeedbackAction, streak: number = 4): boolean {
    if (this.commandHistory.length < streak) return false;
    const recent = this.commandHistory.slice(-streak);
    return recent.every(a => a === action);
  }

  // ---------------------------------------------------------------------------
  // Helpers privados
  // ---------------------------------------------------------------------------

  /**
   * Función sigmoide parametrizada.
   *
   *   σ(x, k) = 1 / (1 + e^{−k·x})
   *
   * Propiedades relevantes para feedback EEG:
   *   σ(0, k)    = 0.5          (punto medio = baseline del paciente)
   *   σ(1, 2)    ≈ 0.88         (1σ por encima → 88 % de intensidad)
   *   σ(1.5, 2)  ≈ 0.95
   *   σ(−1.5, 2) ≈ 0.05         (1.5σ por debajo → casi sin estímulo)
   *   lim x→+∞   = 1.0          (saturación suave, no clipping abrupto)
   *
   * @param x Entrada (z-score o −z-score según zona)
   * @param k Factor de pendiente (k=2 para neurofeedback EEG)
   * @returns  Valor en (0, 1)
   */
  private sigmoid(x: number, k: number): number {
    return 1 / (1 + Math.exp(-k * x));
  }

  /** Mantiene el historial circular de los últimos `historySize` comandos */
  private _recordHistory(action: FeedbackAction): void {
    this.commandHistory.push(action);
    if (this.commandHistory.length > this.historySize) {
      this.commandHistory.shift();
    }
  }
}
