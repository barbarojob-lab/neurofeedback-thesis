/**
 * band-power.ts  —  backend/src/dsp/
 *
 * Extracción de potencia por banda de frecuencia EEG y cálculo del ratio
 * theta/beta, el biomarcador espectral principal para neurofeedback de
 * relajación/trance.
 *
 * ── Relevancia de cada banda para el estado de trance/relajación ─────────
 *
 *  DELTA (1–4 Hz):
 *    Asociado con sueño profundo (NREM3/4) y estados de inconsciencia.
 *    En adultos despiertos: marcador de somnolencia extrema o patología
 *    (encefalopatía, lesión focal). En estados de trance profundo (hipnosis
 *    Ericksonian stage 3) se observa un incremento delta fronto-central
 *    que se correlaciona con disociación cognitiva.
 *    Referencia: Gruzelier, 2009, "A theory of alpha/theta neurofeedback."
 *
 *  THETA (4–8 Hz):
 *    LA banda de mayor interés para neurofeedback de trance/relajación.
 *    Fisiología: generado principalmente por el hipocampo y corteza
 *    cingulada anterior durante: ensoñación diurna (mind-wandering),
 *    meditación profunda, hipnosis, estados de relajación profunda, y
 *    fase hipnagógica del sueño (onset de sueño).
 *    Neurofeedback theta en Fz/Pz → inducción de estados de conciencia
 *    alterada sin pérdida de control voluntario (Peniston & Kulkosky, 1989).
 *    Theta elevado + alpha elevado = marcador de "flow state" (Csikszentmihalyi).
 *
 *  ALPHA (8–12 Hz):
 *    Ritmo de "relajación alerta": ojos cerrados, mente tranquila, sin
 *    procesamiento activo de información. Máximo en occipital (O1/O2) con
 *    ojos cerrados (fenómeno de Berger, 1929). Se atenúa con apertura
 *    ocular o esfuerzo cognitivo (alpha blocking).
 *    Alpha peak frequency (APF, típicamente 10 Hz): marcador de velocidad
 *    de procesamiento cognitivo. APF más alta → respuesta más rápida.
 *    En meditación mindfulness: alpha sincronizado y generalizado.
 *    Para neurofeedback: incremento de alpha = relajación sin somnolencia.
 *
 *  BETA (12–30 Hz):
 *    Estado de alerta activo, concentración, procesamiento cognitivo
 *    y actividad motora. Beta bajo (12–15 Hz, SMR - sensorimotor rhythm):
 *    relajación enfocada, atención sostenida, control inhibitorio.
 *    Beta medio (15–20 Hz): cognición activa, resolución de problemas.
 *    Beta alto (20–30 Hz): ansiedad, rumiación, hiperactivación.
 *    En neurofeedback: beta ELEVADO es indicador de arousal excesivo;
 *    REDUCIR beta → relajación. El ratio theta/beta captura exactamente
 *    este balance: trance/relajación = theta↑ / beta↓.
 *
 *  GAMMA (30–45 Hz) — SOLO USO OFFLINE / POST-SESIÓN:
 *    Representa integración sensorial de alto nivel, binding perceptual,
 *    y conciencia consciente. Potencialmente interesante para estados de
 *    trance avanzado (algunos estudios reportan gamma en meditadores expertos).
 *    ⚠️  POR QUÉ NO SE USA EN FEEDBACK EN TIEMPO REAL:
 *    El EMG (electromiografía) muscular facial genera señal en exactamente
 *    el mismo rango espectral (20–200 Hz, máxima densidad 30–60 Hz).
 *    Los electrodos frontales (Fp1, Fp2, Fz) son los más contaminados.
 *    Separar gamma EEG de EMG facial en tiempo real requiere ICA (Independent
 *    Component Analysis), que no es causal y tiene latencia de segundos.
 *    Usar gamma como señal contingente de feedback generaría refuerzo de
 *    artefactos musculares, no de actividad neuronal real.
 *    Referencia: Thibault et al., NeuroImage 2018.
 *
 * ── Ratio Theta/Beta ─────────────────────────────────────────────────────
 *
 *   TBR = P_theta / P_beta   (razón de potencias absolutas en µV²)
 *
 *   TBR alto (> 3.0) → relajación profunda, posible somnolencia, trance
 *   TBR bajo (< 1.5) → alerta activo, ansiedad, hiperactivación cognitiva
 *   TBR normal (1.5–3.0) → estado de reposo tranquilo
 *
 *   Uso clínico: diagnóstico de TDAH (TBR > 3 en Cz → marcador Lubar, 1991),
 *   monitoreo de anestesia, evaluación de meditación/trance.
 *
 *   Para neurofeedback de relajación el objetivo es incrementar TBR:
 *   theta↑ indica transición hacia estados hipnagógicos/meditativos,
 *   beta↓ indica disminución de rumiación y arousal cortical.
 */

// ---------------------------------------------------------------------------
// Tipos
// ---------------------------------------------------------------------------

/** Potencias por banda en µV² (resultado de integración espectral) */
export interface BandPowers {
  /** Delta 1–4 Hz [µV²]: sueño profundo / disociación en trance */
  delta: number;
  /** Theta 4–8 Hz [µV²]: relajación profunda, meditación, hipnosis */
  theta: number;
  /** Alpha 8–12 Hz [µV²]: relajación alerta, ojos cerrados */
  alpha: number;
  /** Beta 12–30 Hz [µV²]: activación cortical, cognición activa */
  beta: number;
  /**
   * Gamma 30–45 Hz [µV²]: integración sensorial.
   * ⚠️  Solo uso offline — contaminado por EMG en tiempo real.
   */
  gamma: number;
  /** Timestamp de la época analizada (ms desde epoch Unix) */
  timestamp: number;
}

/** Resultado del ratio theta/beta con interpretación clínica */
export interface ThetaBetaResult {
  /** Ratio θ/β = P_theta / P_beta (adimensional) */
  ratio: number;
  /** Potencia theta en µV² */
  thetaPower: number;
  /** Potencia beta en µV² */
  betaPower: number;
  /** Clasificación del estado según el ratio */
  state: "hyperactive" | "alert" | "relaxed" | "drowsy" | "trance";
  /** Descripción legible del estado */
  stateDescription: string;
}

// ---------------------------------------------------------------------------
// Definición de bandas de frecuencia
// ---------------------------------------------------------------------------

interface BandDefinition {
  name: keyof Omit<BandPowers, "timestamp">;
  lowHz: number;
  highHz: number;
}

const EEG_BANDS: BandDefinition[] = [
  { name: "delta", lowHz: 1,  highHz: 4  },
  { name: "theta", lowHz: 4,  highHz: 8  },
  { name: "alpha", lowHz: 8,  highHz: 12 },
  { name: "beta",  lowHz: 12, highHz: 30 },
  { name: "gamma", lowHz: 30, highHz: 45 },
];

// Umbrales del ratio theta/beta para clasificación de estado
const TBR_THRESHOLDS = {
  hyperactive: 1.0,  // TBR < 1.0 → hiperactivación / ansiedad
  alert:       1.5,  // 1.0 ≤ TBR < 1.5 → alerta activo
  relaxed:     3.0,  // 1.5 ≤ TBR < 3.0 → relajación normal
  drowsy:      5.0,  // 3.0 ≤ TBR < 5.0 → somnolencia / trance ligero
                     // TBR ≥ 5.0 → trance profundo / hipnagógico
};

// ---------------------------------------------------------------------------
// Clase principal
// ---------------------------------------------------------------------------

export class BandPowerExtractor {
  /** Resolución espectral en Hz/bin */
  private readonly binResolution: number;

  /** Número de bins útiles (= windowSize / 2) */
  private readonly numBins: number;

  /**
   * @param sampleRate Frecuencia de muestreo en Hz (default 250)
   * @param windowSize Tamaño de la ventana FFT (default 256)
   *
   *   binResolution = sampleRate / windowSize
   *   A 250 sps / 256 pts: Δf ≈ 0.977 Hz (resolución suficiente para EEG)
   */
  constructor(
    public readonly sampleRate: number = 250,
    public readonly windowSize: number = 256
  ) {
    this.binResolution = sampleRate / windowSize;
    this.numBins       = windowSize / 2;
  }

  // ---------------------------------------------------------------------------
  // API pública
  // ---------------------------------------------------------------------------

  /**
   * Extrae la potencia en µV² para cada banda EEG a partir del espectro
   * de magnitudes en dB producido por FFTAnalyzer.analyze().
   *
   * @param magnitudes Float32Array[numBins] en dB de FFTAnalyzer
   * @returns          BandPowers con potencias en µV² por banda
   */
  extract(magnitudes: Float32Array): BandPowers {
    const powers: Partial<BandPowers> = { timestamp: Date.now() };

    for (const band of EEG_BANDS) {
      powers[band.name] = this.bandPowerToUV2(
        magnitudes,
        band.lowHz,
        band.highHz
      );
    }

    return powers as BandPowers;
  }

  /**
   * Calcula el ratio theta/beta y clasifica el estado cognitivo/de trance.
   *
   * Interpretación del TBR para neurofeedback de relajación:
   *   - TBR < 1.0  : hiperactivación, ansiedad (beta dominante)
   *   - TBR 1.0–1.5: alerta activo (cognición, resolución de problemas)
   *   - TBR 1.5–3.0: relajación normal (estado de reposo tranquilo)
   *   - TBR 3.0–5.0: relajación profunda / trance ligero / somnolencia
   *   - TBR > 5.0  : trance profundo / estado hipnagógico
   *
   * @param bands BandPowers de extract()
   * @returns     ThetaBetaResult con ratio, estado y descripción
   */
  computeThetaBetaRatio(bands: BandPowers): ThetaBetaResult {
    const thetaPower = bands.theta;
    const betaPower  = bands.beta;

    // Evitar división por cero: si beta < ε, ratio → Infinity
    // En la práctica, beta siempre tiene algo de potencia en señal EEG real.
    const ratio = betaPower > 1e-12
      ? thetaPower / betaPower
      : thetaPower > 0 ? Infinity : 0;

    const { state, stateDescription } = this._classifyTBR(ratio);

    return { ratio, thetaPower, betaPower, state, stateDescription };
  }

  /**
   * Integra la potencia espectral en µV² para una banda de frecuencias.
   *
   * ── Método de integración ────────────────────────────────────────────
   *
   *   Las magnitudes en dB representan amplitud (no potencia). Para obtener
   *   potencia espectral en µV² se aplica la conversión inversa:
   *
   *     mag_linear = 10^(dB / 20)   [µV — amplitud]
   *     potencia_k = mag_linear²    [µV²/bin]
   *
   *   Luego se suman las potencias de todos los bins dentro de la banda:
   *
   *     P_banda = Σ_{k: f_low ≤ f(k) < f_high} mag_linear(k)²
   *
   *   Nota: se usa suma (no integración trapezoidal) porque los bins tienen
   *   anchura uniforme Δf = binResolution → la suma es equivalente a la
   *   integral rectangular, que es suficientemente precisa para Δf < 1 Hz.
   *
   *   Alternativa más precisa (Welch's PSD con densidad espectral de potencia):
   *     PSD[k] = |X[k]|² / (fs · Σw²)   donde Σw² = suma de coeficientes Hann²
   *   Esto normalizaría por el ancho de banda efectivo de la ventana.
   *   No se implementa aquí para mantener la interfaz simple; usar si se
   *   necesitan comparaciones cuantitativas entre sesiones o sujetos.
   *
   * @param magnitudes  Float32Array[numBins] en dB de FFTAnalyzer.analyze()
   * @param freqLow     Frecuencia inferior de la banda [Hz] (inclusive)
   * @param freqHigh    Frecuencia superior de la banda [Hz] (exclusive)
   * @returns           Potencia integrada en µV²
   */
  bandPowerToUV2(
    magnitudes: Float32Array,
    freqLow: number,
    freqHigh: number
  ): number {
    // Convertir frecuencias a índices de bin (con clipping a [0, numBins))
    const binLow  = Math.max(0,           Math.round(freqLow  / this.binResolution));
    const binHigh = Math.min(this.numBins, Math.round(freqHigh / this.binResolution));

    let sumPower = 0;

    for (let k = binLow; k < binHigh; k++) {
      // Conversión dB → amplitud lineal (µV):
      //   mag_µV = 10^(dB/20)
      //   Nota: 10^(dB/20) y no 10^(dB/10) porque las magnitudes son de amplitud.
      const magLinear = Math.pow(10, magnitudes[k]! / 20);

      // Potencia en µV² por bin
      sumPower += magLinear * magLinear;
    }

    return sumPower;
  }

  // ---------------------------------------------------------------------------
  // Métodos privados
  // ---------------------------------------------------------------------------

  private _classifyTBR(ratio: number): {
    state: ThetaBetaResult["state"];
    stateDescription: string;
  } {
    if (ratio < TBR_THRESHOLDS.hyperactive) {
      return {
        state: "hyperactive",
        stateDescription:
          "Hiperactivación cortical: beta dominante, posible ansiedad o rumiación activa.",
      };
    }
    if (ratio < TBR_THRESHOLDS.alert) {
      return {
        state: "alert",
        stateDescription:
          "Alerta activo: procesamiento cognitivo, atención focalizada.",
      };
    }
    if (ratio < TBR_THRESHOLDS.relaxed) {
      return {
        state: "relaxed",
        stateDescription:
          "Relajación normal: estado de reposo tranquilo, mente calmada.",
      };
    }
    if (ratio < TBR_THRESHOLDS.drowsy) {
      return {
        state: "drowsy",
        stateDescription:
          "Relajación profunda / trance ligero: theta dominante, posible somnolencia.",
      };
    }
    return {
      state: "trance",
      stateDescription:
        "Trance profundo / estado hipnagógico: theta muy elevado, disociación parcial.",
    };
  }

  /**
   * Retorna un resumen de las potencias con interpretación por banda.
   * Útil para logging de sesión y reportes clínicos.
   */
  summarize(bands: BandPowers, tbr: ThetaBetaResult): string {
    const fmt = (v: number) => v.toFixed(4);
    return [
      `── BandPowers [µV²] @ ${new Date(bands.timestamp).toISOString()} ──`,
      `  Delta (1–4 Hz):  ${fmt(bands.delta).padStart(10)} µV²  (somnolencia/trance profundo)`,
      `  Theta (4–8 Hz):  ${fmt(bands.theta).padStart(10)} µV²  ← señal de trance/relajación`,
      `  Alpha (8–12 Hz): ${fmt(bands.alpha).padStart(10)} µV²  (relajación alerta)`,
      `  Beta (12–30 Hz): ${fmt(bands.beta).padStart(10)} µV²  (activación cognitiva)`,
      `  Gamma (30–45 Hz):${fmt(bands.gamma).padStart(10)} µV²  [solo offline — EMG contaminado]`,
      `  Theta/Beta ratio: ${tbr.ratio.toFixed(3)}  →  ${tbr.state}: ${tbr.stateDescription}`,
    ].join("\n");
  }
}
