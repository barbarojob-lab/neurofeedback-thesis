/**
 * fft-analyzer.ts  —  backend/src/dsp/
 *
 * Encapsula el análisis espectral mediante FFT de Cooley-Tukey (radix-2).
 * Usa la librería 'fft.js' (O(N log N), pura JS, sin dependencias nativas).
 *
 * ── Flujo de procesamiento ────────────────────────────────────────────────
 *
 *   EEG samples (µV)
 *       ↓  SlidingWindow.applyHannWindow()
 *   Ventana Hann aplicada
 *       ↓  FFTAnalyzer.analyze()
 *   Espectro complejo [re₀, im₀, re₁, im₁, …]   (fft.js — formato intercalado)
 *       ↓  magnitud = √(re² + im²) / N  →  dB = 20·log10(mag)
 *   Float32Array[N/2] magnitudes en dB (solo mitad positiva del espectro)
 *       ↓  BandPowerExtractor.extract()
 *   BandPowers { delta, theta, alpha, beta, gamma }
 *
 * ── Normalización de magnitudes ──────────────────────────────────────────
 *
 *   La FFT produce magnitudes proporcionales al tamaño de ventana N.
 *   División por N → amplitud en µV independiente de N.
 *   Conversión a dB: 20·log10(mag/N + ε), ε=1e-10 evita log10(0).
 *
 *   Referencia: Proakis & Manolakis, "Digital Signal Processing", §8.1.
 *
 * ── Por qué solo N/2 bins ────────────────────────────────────────────────
 *
 *   Para señal real x[n], la FFT tiene simetría hermítica: X[k]=conj(X[N-k]).
 *   Los bins 0…N/2 contienen toda la información no redundante.
 *   Bins N/2+1…N-1 son espejos → se descartan.
 *
 *   Resolución espectral: Δf = fs / N
 *   Frecuencia del bin k: f(k) = k · fs / N
 */

// ---------------------------------------------------------------------------
// Importación de fft.js
// ---------------------------------------------------------------------------

// fft.js exporta una clase constructora por defecto (CJS).
// Se declara la interfaz local para tipado sin depender de @types/fft.js.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const FFT = require("fft.js") as new (size: number) => FFTInstance;

interface FFTInstance {
  /** Crea array complejo intercalado pre-allocado [re,im,re,im,...] */
  createComplexArray(): number[];
  /** Transforma array real → espectro complejo (mitad positiva) */
  realTransform(out: number[], input: ArrayLike<number>): void;
  /** Completa la simetría hermítica si se necesita el espectro completo */
  completeSpectrum(spectrum: number[]): void;
}

// ---------------------------------------------------------------------------
// Clase principal
// ---------------------------------------------------------------------------

export class FFTAnalyzer {
  /** Instancia reutilizable de fft.js — sin re-allocación por epoch */
  private readonly fft: FFTInstance;

  /**
   * Buffer de salida compleja reutilizado entre llamadas a analyze().
   * Formato intercalado: [re₀, im₀, re₁, im₁, …, re_{N-1}, im_{N-1}]
   * Longitud total: 2 × windowSize
   */
  private readonly complexOut: number[];

  /** Número de bins espectrales útiles = windowSize / 2 */
  public readonly numBins: number;

  /** Resolución espectral Δf = sampleRate / windowSize [Hz/bin] */
  public readonly binResolution: number;

  /**
   * @param windowSize Tamaño de la ventana FFT: 256 o 512.
   *                   Debe coincidir con SlidingWindow.windowSize.
   * @param sampleRate Frecuencia de muestreo en Hz (default 250).
   *
   *   A 250 sps, windowSize=256 → 128 bins útiles, Δf ≈ 0.977 Hz
   *   A 250 sps, windowSize=512 → 256 bins útiles, Δf ≈ 0.488 Hz
   */
  constructor(
    public readonly windowSize: 256 | 512,
    public readonly sampleRate: number = 250
  ) {
    this.fft           = new FFT(windowSize);
    this.complexOut    = this.fft.createComplexArray();
    this.numBins       = windowSize / 2;
    this.binResolution = sampleRate / windowSize;
  }

  // ---------------------------------------------------------------------------
  // Métodos públicos
  // ---------------------------------------------------------------------------

  /**
   * Ejecuta la FFT sobre datos ya ventaneados y retorna magnitudes en dB.
   *
   * @param windowedData Float32Array[windowSize] multiplicado por Hann
   * @returns            Float32Array[windowSize/2] en dB, bins 0…N/2-1
   *                     (frecuencias 0 Hz … fs/2 Hz)
   */
  analyze(windowedData: Float32Array): Float32Array {
    if (windowedData.length !== this.windowSize) {
      throw new RangeError(
        `analyze: datos[${windowedData.length}] ≠ windowSize(${this.windowSize})`
      );
    }

    // ── FFT real → compleja ─────────────────────────────────────────────
    // realTransform calcula solo la mitad positiva del espectro (bins 0…N/2).
    // No es necesario completeSpectrum porque solo usamos la mitad positiva.
    this.fft.realTransform(this.complexOut, windowedData);

    // ── Cálculo de magnitudes en dB ──────────────────────────────────────
    const magnitudes = new Float32Array(this.numBins);
    const N          = this.windowSize;

    for (let k = 0; k < this.numBins; k++) {
      // Formato intercalado: índice real = 2k, imaginario = 2k+1
      const re = this.complexOut[2 * k]!;
      const im = this.complexOut[2 * k + 1]!;

      // Magnitud lineal normalizada:
      //   |X[k]| / N  →  amplitud en µV, independiente del tamaño de ventana
      const mag = Math.sqrt(re * re + im * im) / N;

      // Conversión a dB de amplitud (factor 20, no 10):
      //   20·log10 se usa para amplitud (V, µV);
      //   10·log10 se usa para potencia (W, µV²).
      //   ε = 1e-10 µV → piso de ruido numérico ≈ −200 dB
      magnitudes[k] = 20 * Math.log10(mag + 1e-10);
    }

    return magnitudes;
  }

  /**
   * Convierte índice de bin → frecuencia en Hz.
   *
   *   f(k) = k · fs / N  =  k · binResolution
   *
   * @param binIndex Índice [0 … numBins−1]
   * @returns        Frecuencia central del bin en Hz
   */
  getBinFrequency(binIndex: number): number {
    return binIndex * this.binResolution;
  }

  /**
   * Magnitud espectral en una frecuencia arbitraria mediante interpolación
   * lineal entre los dos bins más cercanos.
   *
   * Útil para monitorear picos específicos (e.g., alfa peak ~10 Hz)
   * que no caen exactamente en un bin de la FFT.
   *
   * @param magnitudes Float32Array[numBins] de analyze()
   * @param freqHz     Frecuencia objetivo [0 … fs/2] Hz
   * @returns          Magnitud interpolada en dB
   */
  getMagnitudeAtFrequency(magnitudes: Float32Array, freqHz: number): number {
    const nyquist = this.sampleRate / 2;

    if (freqHz < 0 || freqHz > nyquist) {
      throw new RangeError(
        `Frecuencia ${freqHz} Hz fuera del rango [0, ${nyquist}] Hz`
      );
    }

    // Bin continuo → bins enteros vecinos + fracción para interpolación
    const exactBin = freqHz / this.binResolution;
    const loBin    = Math.floor(exactBin);
    const hiBin    = Math.min(loBin + 1, this.numBins - 1);
    const frac     = exactBin - loBin;

    // Interpolación lineal (válida para espectros suaves post-ventana Hann)
    return magnitudes[loBin]! * (1 - frac) + magnitudes[hiBin]! * frac;
  }

  /**
   * Índice del bin más cercano a una frecuencia dada.
   *
   * @param freqHz Frecuencia en Hz
   * @returns      Índice de bin entero
   */
  freqToBin(freqHz: number): number {
    return Math.round(freqHz / this.binResolution);
  }
}
