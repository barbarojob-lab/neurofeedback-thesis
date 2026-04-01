/**
 * running-zscore.ts  —  backend/src/adaptive/
 *
 * Normalización adaptativa en tiempo real para señales EEG de neurofeedback.
 * Convierte valores absolutos de potencia espectral (e.g., ratio theta/beta
 * en µV²) en z-scores relativos al baseline dinámico del paciente en sesión.
 *
 * ── Por qué necesitamos normalización adaptativa ──────────────────────────
 *
 *   La potencia EEG absoluta varía enormemente entre pacientes (~10×) y entre
 *   sesiones del mismo paciente (impedancia de electrodos, nivel de hidratación,
 *   hora del día). Un umbral fijo para "theta alto" que funcione para un sujeto
 *   puede ser completamente inútil para otro.
 *
 *   El z-score deslizante resuelve esto: en lugar de comparar con valores
 *   absolutos, cada muestra se evalúa contra la distribución reciente del
 *   propio paciente. z=+2 significa "2 desviaciones estándar por encima de
 *   tu propio baseline de los últimos 3 minutos", independientemente de si
 *   ese baseline es 0.5 µV² o 5 µV².
 *
 * ── Por qué Welford es superior al cálculo naive de varianza ─────────────
 *
 *   MÉTODO NAIVE (incorrecto para uso clínico):
 *     var = (Σx²)/n − (Σx/n)²
 *
 *   Problema: requiere almacenar Σx y Σx² como acumuladores.
 *   Con señales EEG de alta frecuencia durante 3 minutos a 250 sps:
 *     n ≈ 45.000 samples → Σx² puede alcanzar valores de 10^10.
 *   La diferencia entre dos números grandes casi iguales produce
 *   CANCELACIÓN CATASTRÓFICA en punto flotante IEEE 754 de 64 bits:
 *
 *     Ejemplo: Σx²/n = 1.000000001, (Σx/n)² = 1.000000000
 *     → varianza = 0.000000001 (correcto)
 *     Pero si hay error de redondeo en el 15º dígito → varianza = -0.000000003
 *     → Varianza NEGATIVA → std = NaN → z-score indefinido → feedback incorrecto
 *
 *   Esto no es teórico: el error se manifiesta especialmente cuando la señal
 *   tiene poca variabilidad (paciente en estado estable de relajación),
 *   exactamente la condición más importante para este sistema.
 *
 *   ALGORITMO DE WELFORD (Welford, 1962 — usado en GNU GSL, NumPy, SciPy):
 *     Para cada nueva muestra xₙ:
 *       δ₁ = xₙ − μₙ₋₁          (desviación respecto a media ANTES)
 *       μₙ = μₙ₋₁ + δ₁/n        (media actualizada)
 *       δ₂ = xₙ − μₙ            (desviación respecto a media DESPUÉS)
 *       M₂ₙ = M₂ₙ₋₁ + δ₁·δ₂    (suma de cuadrados de desviaciones)
 *       σ²ₙ = M₂ₙ / (n−1)       (varianza muestral insesgada)
 *
 *   Ventajas:
 *   1. Opera sobre desviaciones (números pequeños), nunca sobre sumas de cuadrados.
 *   2. Numéricamente estable para cualquier magnitud de señal.
 *   3. Exactamente una pasada (online) — sin necesidad de almacenar el historial.
 *   4. Downdate exacto al eliminar muestras antiguas de la ventana deslizante
 *      (usando la formulación de Chan et al., 1979 para combinación/substracción).
 *
 *   VENTANA DESLIZANTE + WELFORD:
 *   Al eliminar la muestra más antigua (x_old) de la ventana se aplica el
 *   downdate de Welford:
 *     n' = n − 1
 *     δ_old = x_old − μ_old
 *     μ' = (n·μ − x_old) / n'
 *     δ'_old = x_old − μ'
 *     M₂' = M₂ − δ_old · δ'_old
 *
 *   Esta es la única forma numéricamente segura de implementar una varianza
 *   deslizante. El método alternativo de "restar el valor antiguo" del
 *   acumulador naive introduce errores acumulativos que crecen con el tiempo.
 *
 *   Referencia: Welford, B.P. (1962). "Note on a method for calculating
 *   corrected sums of squares and products." Technometrics 4(3):419-420.
 *
 * ── Suavizado EMA sobre el z-score ───────────────────────────────────────
 *
 *   El z-score raw tiene variabilidad alta sample-a-sample (EEG es ruidoso
 *   incluso tras filtrado). Aplicar una EMA con α=0.1 suaviza las
 *   transiciones del feedback sin añadir latencia significativa:
 *
 *     z_smooth[n] = α·z_raw[n] + (1−α)·z_smooth[n−1]
 *
 *   Con α=0.1: constante de tiempo τ ≈ 1/α = 10 epochs.
 *   A 4 epochs/segundo (hopSize=64, fs=250): τ ≈ 2.5 segundos.
 *   Suficiente para transiciones suaves de feedback visual/auditivo
 *   sin lag perceptible para el terapeuta.
 */

// ---------------------------------------------------------------------------
// Tipos
// ---------------------------------------------------------------------------

export interface ZScoreResult {
  /** Z-score raw (sin suavizar): cuántas σ está la muestra sobre la media */
  zRaw: number;
  /** Z-score suavizado con EMA α=0.1 — usar este para feedback */
  zSmooth: number;
  /** Media actual de la ventana deslizante */
  mean: number;
  /** Desviación estándar actual de la ventana deslizante */
  std: number;
  /** Número de muestras en la ventana activa */
  n: number;
  /** true si hay suficientes muestras para un z-score confiable (n ≥ 30) */
  isReady: boolean;
}

/** Entrada del buffer circular con timestamp para expiración por tiempo */
interface TimestampedSample {
  value: number;
  timestamp: number; // ms Unix
}

// ---------------------------------------------------------------------------
// Clase principal
// ---------------------------------------------------------------------------

export class RunningZScore {
  // ── Parámetros de la ventana ─────────────────────────────────────────────
  /** Tamaño máximo del buffer en número de muestras */
  private readonly maxSamples: number;

  /** Duración de la ventana en milisegundos */
  private readonly windowMs: number;

  // ── Buffer circular con timestamps ──────────────────────────────────────
  private readonly buffer: TimestampedSample[];
  private head   = 0; // índice de escritura (circular)
  private count  = 0; // número de elementos válidos en el buffer
  private tail   = 0; // índice de lectura (más antiguo)

  // ── Estado de Welford ────────────────────────────────────────────────────
  /** Media acumulada μₙ */
  private mean   = 0;
  /** Suma de cuadrados de desviaciones M₂ (= (n-1)·σ²) */
  private m2     = 0;
  /** Número de muestras procesadas en el estado Welford actual */
  private n      = 0;

  // ── EMA sobre z-score ───────────────────────────────────────────────────
  /** Factor de suavizado EMA: α=0.1 → τ ≈ 10 epochs */
  private readonly emaAlpha = 0.1;
  /** Valor EMA actual (inicializado en 0 hasta primera muestra) */
  private zEma   = 0;
  /** true tras la primera actualización EMA */
  private emaInit = false;

  /**
   * @param windowMinutes  Duración de la ventana deslizante en minutos (default 3)
   * @param maxSamplesHint Máximo de muestras esperadas en la ventana.
   *                       Default: windowMinutes × 60 × 4 epochs/seg.
   *                       Ajustar según hopSize / sampleRate del pipeline.
   */
  constructor(
    public readonly windowMinutes: number = 3,
    maxSamplesHint?: number
  ) {
    this.windowMs   = windowMinutes * 60 * 1000;
    // Buffer pre-allocado: evitar resize dinámico en el hot path.
    // Estimación conservadora: 4 epochs/s × 60 s/min × windowMinutes × 1.2 (margen)
    this.maxSamples = maxSamplesHint ?? Math.ceil(windowMinutes * 60 * 4 * 1.2);
    this.buffer     = new Array<TimestampedSample>(this.maxSamples);

    // Pre-inicializar slots para evitar undefined en acceso circular
    for (let i = 0; i < this.maxSamples; i++) {
      this.buffer[i]! = { value: 0, timestamp: 0 };
    }
  }

  // ---------------------------------------------------------------------------
  // API pública
  // ---------------------------------------------------------------------------

  /**
   * Ingresa un nuevo valor (e.g., ratio theta/beta del epoch actual),
   * actualiza las estadísticas de Welford y retorna el z-score suavizado.
   *
   * Complejidad amortizada: O(k) donde k = número de muestras expiradas
   * en este ciclo (típicamente 0 o 1). En estado estacionario: O(1).
   *
   * @param value Valor escalar a normalizar (e.g., TBR ratio, potencia theta)
   * @returns     ZScoreResult con z-score raw, suavizado y estadísticas
   */
  push(value: number): ZScoreResult {
    const now = Date.now();

    // ── 1. Expirar muestras antiguas (downdate de Welford) ───────────────
    this._expireOldSamples(now);

    // ── 2. Insertar nueva muestra en buffer circular ─────────────────────
    this.buffer[this.head]! = { value, timestamp: now };
    this.tail = (this.tail + 1) % this.maxSamples;
    this.head = (this.head + 1) % this.maxSamples;
    if (this.count < this.maxSamples) this.count++;

    // ── 3. Update de Welford (updte con nueva muestra) ───────────────────
    this.n++;
    const delta1 = value - this.mean;          // δ₁ = x − μ_anterior
    this.mean   += delta1 / this.n;            // μ_nueva = μ + δ₁/n
    const delta2 = value - this.mean;          // δ₂ = x − μ_nueva
    this.m2     += delta1 * delta2;            // M₂ += δ₁·δ₂  (Welford core)

    // ── 4. Calcular z-score ──────────────────────────────────────────────
    const std    = this._currentStd();
    const zRaw   = this._computeZ(value, this.mean, std);

    // ── 5. Suavizado EMA ─────────────────────────────────────────────────
    if (!this.emaInit) {
      this.zEma   = zRaw;
      this.emaInit = true;
    } else {
      // z_ema[n] = α·z_raw[n] + (1−α)·z_ema[n−1]
      this.zEma = this.emaAlpha * zRaw + (1 - this.emaAlpha) * this.zEma;
    }

    return {
      zRaw,
      zSmooth : this.zEma,
      mean    : this.mean,
      std,
      n       : this.n,
      isReady : this.n >= 30, // mínimo estadístico para z-score confiable
    };
  }

  /**
   * Resetea todo el estado interno.
   * Llamar al inicio de una nueva sesión o al cambiar de paciente.
   */
  reset(): void {
    this.head    = 0;
    this.count   = 0;
    this.mean    = 0;
    this.m2      = 0;
    this.n       = 0;
    this.zEma    = 0;
    this.emaInit = false;

    // Limpiar timestamps del buffer para que _expireOldSamples no procese
    // entradas de la sesión anterior

    for (let i = 0; i < this.maxSamples; i++) {
      (this.buffer[i] as TimestampedSample).timestamp = 0;
    }

  }

  /**
   * Retorna las estadísticas actuales de la ventana.
   * Útil para el panel del terapeuta y logging de sesión.
   */
  getStats(): {
    mean: number;
    std: number;
    n: number;
    windowSize: number;
    windowMinutes: number;
    isReady: boolean;
  } {
    return {
      mean          : this.mean,
      std           : this._currentStd(),
      n             : this.n,
      windowSize    : this.maxSamples,
      windowMinutes : this.windowMinutes,
      isReady       : this.n >= 30,
    };
  }

  // ---------------------------------------------------------------------------
  // Métodos privados
  // ---------------------------------------------------------------------------

  /**
   * Expira muestras cuyo timestamp es anterior al límite de la ventana.
   * Aplica el downdate de Welford (Chan et al., 1979) por cada muestra
   * eliminada para mantener M₂ numéricamente exacto.
   *
   * Downdate de Welford al eliminar x_old de una ventana de n elementos:
   *   n'    = n − 1
   *   μ'    = (n·μ − x_old) / n'
   *   M₂'   = M₂ − (x_old − μ) · (x_old − μ')
   *
   * Equivalente a "revertir" el update de Welford. Esta es la operación
   * que hace imposible la alternativa naive: con acumuladores Σx y Σx²,
   * la substracción de valores antiguos produce errores numéricos
   * acumulativos que crecen monotónicamente con el tiempo de sesión.
   */
  private _expireOldSamples(now: number): void {
    const cutoff = now - this.windowMs;

    // El puntero de "lectura" del buffer circular comienza donde termina
    // el puntero de escritura (el elemento más antiguo en un buffer lleno)
    let readPtr = this.count < this.maxSamples
      ? 0                                        // buffer no lleno: leer desde 0
      : this.tail;                               // buffer lleno: el más antiguo es tail

    let remaining = this.count;

    while (remaining > 0) {
      const idx = readPtr % this.maxSamples;
      const slot = this.buffer[idx] as TimestampedSample;

      // Parar en cuanto encontremos una muestra dentro de la ventana
      if (slot.timestamp === 0 || slot.timestamp > cutoff) break;

      // ── Downdate de Welford ──────────────────────────────────────────
      if (this.n > 1) {
        const xOld    = slot.value;
        const nNew    = this.n - 1;
        const meanNew = (this.n * this.mean - xOld) / nNew;

        // M₂' = M₂ − (x_old − μ_old)·(x_old − μ_new)
        this.m2  -= (xOld - this.mean) * (xOld - meanNew);
        this.m2   = Math.max(0, this.m2); // guardia numérica: M₂ ≥ 0 siempre
        this.mean = meanNew;
        this.n    = nNew;
      } else if (this.n === 1) {
        // Último elemento: volver a estado vacío
        this.mean = 0;
        this.m2   = 0;
        this.n    = 0;
      }

      // Marcar slot como expirado
      slot.timestamp = 0;
      this.count--;
      remaining--;
      readPtr++;
    }
  }

  /** Desviación estándar muestral (con Bessel's correction n−1) */
  private _currentStd(): number {
    if (this.n < 2) return 0;
    // σ = √(M₂ / (n−1))  — varianza muestral insesgada (Bessel's correction)
    return Math.sqrt(this.m2 / (this.n - 1));
  }

  /**
   * Z-score estándar con manejo de std ≈ 0.
   * Si std < ε la señal es perfectamente constante; z=0 (neutro) es
   * más apropiado que Infinity para el motor de feedback.
   */
  private _computeZ(value: number, mean: number, std: number): number {
    if (std < 1e-10) return 0;
    return (value - mean) / std;
  }
}
