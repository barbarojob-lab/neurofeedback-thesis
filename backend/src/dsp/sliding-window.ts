/**
 * sliding-window.ts  —  backend/src/dsp/
 *
 * Buffer circular de tamaño fijo basado en Float32Array pre-allocado.
 * Sin allocaciones dinámicas en el hot path → cero presión sobre el GC
 * de V8, crítico para mantener latencia constante en tiempo real (< 4 ms
 * por epoch a 250 sps).
 *
 * ── Diseño del buffer circular ────────────────────────────────────────────
 *
 *  Índices:
 *    writeHead   : apunta a la próxima posición de escritura (mod windowSize)
 *    sampleCount : muestras acumuladas desde el último proceso (mod hopSize)
 *
 *  Lectura: getWindow() copia los windowSize samples en orden cronológico
 *           desenrollando el buffer circular → O(2·windowSize) en el peor
 *           caso, con acceso contiguo a TypedArray = muy cache-friendly.
 *
 *  No hay Array.push / Array.slice en el hot path. Todo es indexación directa
 *  en el mismo Float32Array de windowSize elementos.
 *
 * ── Por qué 256 / 512 muestras ───────────────────────────────────────────
 *
 *  A 250 sps:
 *    windowSize=256 → epoch de 1.024 s, resolución espectral Δf ≈ 0.977 Hz
 *    windowSize=512 → epoch de 2.048 s, resolución espectral Δf ≈ 0.488 Hz
 *
 *  Para neurofeedback en tiempo real se prefiere 256 (latencia baja).
 *  Para análisis offline o bandas delta finas se prefiere 512 (mejor Δf).
 *
 *  La resolución espectral mínima para separar delta (1–4 Hz) de theta
 *  (4–8 Hz) es Δf < 1 Hz → ambos tamaños son suficientes a 250 sps.
 */

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

/** Coeficientes Hann precalculados y cacheados por tamaño de ventana */
const HANN_CACHE = new Map<number, Float32Array>();

function getHannCoefficients(size: number): Float32Array {
  if (HANN_CACHE.has(size)) return HANN_CACHE.get(size)!;

  const hann = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    // w(n) = 0.5 · (1 − cos(2π·n / N))   — variante "periódica"
    //
    // Denominador N (no N-1): garantiza continuidad circular perfecta,
    // reduciendo spectral leakage ~6 dB adicionales respecto a la variante
    // simétrica cuando el análisis usa overlap-add (Welch's method).
    hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / size));
  }

  HANN_CACHE.set(size, hann);
  return hann;
}

// ---------------------------------------------------------------------------
// Clase principal
// ---------------------------------------------------------------------------

export class SlidingWindow {
  /** Buffer circular pre-allocado — nunca se reasigna tras construcción */
  private readonly buffer: Float32Array;

  /** Puntero de escritura (avanza módulo windowSize) */
  private writeHead = 0;

  /**
   * Contador de muestras desde el último disparo de proceso.
   * Se resetea a 0 cuando alcanza hopSize.
   */
  private sampleCount = 0;

  /**
   * Total de muestras insertadas desde la creación / último reset.
   * Permite distinguir "buffer aún no lleno" (< windowSize muestras).
   */
  private totalInserted = 0;

  /**
   * @param windowSize Tamaño de la ventana FFT en muestras: 256 o 512.
   *                   Debe ser potencia de 2 para el algoritmo Cooley-Tukey.
   * @param hopSize    Número de muestras nuevas entre análisis consecutivos.
   *
   *   Overlap (%) = (windowSize − hopSize) / windowSize × 100
   *
   *   Configuraciones típicas en neurofeedback:
   *     hopSize = windowSize/4  → 75 % overlap (suavidad de feedback alta)
   *     hopSize = windowSize/2  → 50 % overlap (balance latencia/suavidad)
   *     hopSize = windowSize    →  0 % overlap (análisis de bloques disjuntos)
   *
   *   75 % overlap es el estándar en Welch's PSD para minimizar varianza
   *   del estimador espectral (Heinzel et al., 2002).
   */
  constructor(
    public readonly windowSize: 256 | 512,
    public readonly hopSize: number
  ) {
    if (hopSize < 1 || hopSize > windowSize) {
      throw new RangeError(
        `hopSize debe estar en [1, ${windowSize}], recibido: ${hopSize}`
      );
    }
    this.buffer = new Float32Array(windowSize);
  }

  // ---------------------------------------------------------------------------
  // API pública
  // ---------------------------------------------------------------------------

  /**
   * Inserta una nueva muestra en el buffer circular.
   *
   * Complejidad: O(1) — escritura de un único elemento sin allocaciones.
   *
   * @param sample Muestra EEG filtrada en µV
   * @returns `true` cuando se han acumulado `hopSize` muestras nuevas
   *          Y el buffer tiene al menos `windowSize` muestras históricas.
   *          Solo cuando retorna `true` tiene sentido llamar a `getWindow()`.
   */
  push(sample: number): boolean {
    // Escribir en posición circular y avanzar puntero
    this.buffer[this.writeHead] = sample;
    this.writeHead = (this.writeHead + 1) % this.windowSize;
    this.totalInserted++;
    this.sampleCount++;

    // No disparar proceso hasta que el buffer esté lleno
    const bufferFull = this.totalInserted >= this.windowSize;

    if (this.sampleCount >= this.hopSize) {
      this.sampleCount = 0;
      return bufferFull;
    }

    return false;
  }

  /**
   * Retorna una copia de la ventana actual en orden cronológico.
   *
   * El buffer circular almacena datos en orden de escritura (no lineal).
   * Esta función desenrolla el buffer en dos copias TypedArray:
   *   1. buffer[writeHead … end]  → muestras más antiguas
   *   2. buffer[0 … writeHead-1] → muestras más recientes
   *
   * Se usa TypedArray.set() para ambas copias → operación SIMD en V8,
   * significativamente más rápida que un loop manual.
   *
   * @returns Float32Array[windowSize] en orden temporal [más antiguo → más reciente]
   */
  getWindow(): Float32Array {
    const out  = new Float32Array(this.windowSize);
    const tail = this.buffer.subarray(this.writeHead);             // [writeHead…end]
    const head = this.buffer.subarray(0, this.writeHead);          // [0…writeHead)
    out.set(tail, 0);
    out.set(head, tail.length);
    return out;
  }

  /**
   * Aplica la ventana Hann a los datos de entrada (retorna nueva copia).
   *
   * ── Por qué Hann para EEG ────────────────────────────────────────────────
   *
   * Rectangular (sin ventana):
   *   Spectral leakage severo — discontinuidades en bordes generan lóbulos
   *   laterales de −13 dB que contaminan bandas EEG adyacentes (e.g., alfa
   *   "gotea" sobre theta, distorsionando el ratio theta/beta).
   *
   * Hann (coseno elevado):
   *   Lóbulo lateral máximo: −31.5 dB.
   *   Rolloff: −18 dB/octava.
   *   Suficiente para separar delta/theta/alfa/beta con bins de ~1 Hz.
   *   Ampliamente usado en análisis EEG con método de Welch.
   *
   * Hamming:
   *   Lóbulo lateral: −41.8 dB (mejor supresión), pero discontinuidad en
   *   extremos introduce un escalón DC que puede sesgar la banda delta (1–4 Hz).
   *
   * Blackman:
   *   −58 dB de lóbulo lateral, pero resolución espectral 50 % peor.
   *   Solo justificado si se requiere separar componentes muy próximas
   *   (< 1 Hz de diferencia) — innecesario para las bandas EEG estándar.
   *
   * Conclusión: Hann es el mejor compromiso resolución/leakage para EEG clínico.
   *
   * @param data Float32Array de longitud exacta `windowSize`
   * @returns    Nuevo Float32Array con valores multiplicados por coeficientes Hann
   */
  applyHannWindow(data: Float32Array): Float32Array {
    if (data.length !== this.windowSize) {
      throw new RangeError(
        `applyHannWindow: datos[${data.length}] ≠ windowSize(${this.windowSize})`
      );
    }

    const hann = getHannCoefficients(this.windowSize);
    const out  = new Float32Array(this.windowSize);

    for (let i = 0; i < this.windowSize; i++) {
      out[i] = data[i]! * hann[i]!;
    }

    return out;
  }

  /**
   * Resetea el buffer y todos los contadores al estado inicial.
   * Llamar al inicio de una nueva sesión o epoch.
   */
  reset(): void {
    this.buffer.fill(0);
    this.writeHead     = 0;
    this.sampleCount   = 0;
    this.totalInserted = 0;
  }

  /** Muestras totales insertadas desde la creación / último reset */
  get insertedCount(): number { return this.totalInserted; }

  /** true si el buffer contiene al menos windowSize muestras históricas */
  get isFull(): boolean { return this.totalInserted >= this.windowSize; }
}
