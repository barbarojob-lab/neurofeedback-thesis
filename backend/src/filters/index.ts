/**
 * index.ts  —  backend/src/filters/
 *
 * Punto de entrada único para el módulo de filtros EEG.
 * Re-exporta ambos filtros y expone la fábrica `createEEGFilterChain`.
 *
 * Uso típico en el pipeline de adquisición:
 *
 *   import { createEEGFilterChain } from './filters';
 *
 *   const chain = createEEGFilterChain(250);
 *
 *   // Por cada muestra entrante del ADC:
 *   const filtered = chain.bandpass.process(
 *     chain.notch.process(rawSample)
 *   );
 *
 * Orden de aplicación recomendado:
 *   raw → [Notch 50 Hz] → [Butterworth BP 1–30 Hz] → feature extraction
 *
 *   Razón: el notch elimina primero la interferencia sinusoidal de 50 Hz;
 *   el butterworth después limita la banda sin interactuar con el residuo
 *   de red. Invertir el orden no es incorrecto pero aumenta el ringing
 *   del transitorio inicial al pasar energía de 50 Hz por el BP primero.
 */

// ---------------------------------------------------------------------------
// Re-exportaciones
// ---------------------------------------------------------------------------

export { NotchFilter }          from "./notch-filter";
export type { BiquadCoefficients } from "./notch-filter";

export { ButterworthBandpass }  from "./butterworth-filter";
export type { BiquadSection }   from "./butterworth-filter";

// ---------------------------------------------------------------------------
// Tipos del chain
// ---------------------------------------------------------------------------

import { NotchFilter }         from "./notch-filter";
import { ButterworthBandpass } from "./butterworth-filter";

/**
 * Cadena de filtros EEG lista para usar.
 *
 * - `notch`    : Filtro Notch IIR biquad (elimina 50 Hz / 60 Hz de red eléctrica)
 * - `bandpass` : Filtro Butterworth BP orden 4, SOS (pasa 1–30 Hz por defecto)
 *
 * Llamar a `process(sample)` en el orden correcto:
 *   const y = chain.bandpass.process(chain.notch.process(x));
 *
 * O usar el método de conveniencia `chain.processChain(sample)`.
 */
export interface EEGFilterChain {
  /** Filtro Notch para eliminar interferencia de red eléctrica */
  notch: NotchFilter;
  /** Filtro Butterworth bandpass 1–30 Hz, orden 4 */
  bandpass: ButterworthBandpass;
  /**
   * Aplica la cadena completa (notch → bandpass) a una muestra.
   * Equivalente a: bandpass.process(notch.process(sample))
   *
   * @param sample Muestra EEG cruda en µV
   * @returns      Muestra filtrada en µV
   */
  processChain(sample: number): number;
  /**
   * Resetea el estado interno de ambos filtros.
   * Llamar al inicio de cada epoch / nueva sesión de registro.
   */
  resetChain(): void;
}

// ---------------------------------------------------------------------------
// Fábrica
// ---------------------------------------------------------------------------

/**
 * Crea y retorna una cadena de filtros EEG completamente configurada.
 *
 * @param sampleRate Frecuencia de muestreo del hardware EEG en Hz
 *                   (default 250 Hz — compatible con OpenBCI Cyton, Muse, NeuroSky)
 * @param notchFreq  Frecuencia de la interferencia de red en Hz
 *                   50 Hz (Europa/Asia) | 60 Hz (América del Norte)
 *                   (default 50 Hz)
 * @param lowCut     Frecuencia de corte inferior del bandpass en Hz  (default 1 Hz)
 * @param highCut    Frecuencia de corte superior del bandpass en Hz  (default 30 Hz)
 *
 * @returns EEGFilterChain lista para procesar muestras en tiempo real
 *
 * @example
 * // Pipeline estándar europeo (50 Hz, 250 sps):
 * const chain = createEEGFilterChain(250);
 *
 * // Pipeline norteamericano (60 Hz, 512 sps):
 * const chain = createEEGFilterChain(512, 60);
 *
 * // Pipeline con banda personalizada (e.g. solo delta-theta 0.5–8 Hz):
 * const chain = createEEGFilterChain(250, 50, 0.5, 8);
 */
export function createEEGFilterChain(
  sampleRate: number = 250,
  notchFreq:  number = 50,
  lowCut:     number = 1,
  highCut:    number = 30
): EEGFilterChain {
  const notch    = new NotchFilter(notchFreq, sampleRate);
  const bandpass = new ButterworthBandpass(lowCut, highCut, sampleRate);

  return {
    notch,
    bandpass,

    processChain(sample: number): number {
      // 1. Eliminar interferencia de red eléctrica
      const afterNotch = notch.process(sample);
      // 2. Limitar banda espectral al rango EEG de interés
      return bandpass.process(afterNotch);
    },

    resetChain(): void {
      notch.reset();
      bandpass.reset();
    },
  };
}
