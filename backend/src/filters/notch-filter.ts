/**
 * notch-filter.ts
 * Implementación de filtro Notch IIR biquad de segundo orden
 * usando Direct Form II Transposed y transformación bilineal estándar.
 *
 * Teoría:
 * Un filtro Notch (band-stop) atenúa una frecuencia específica (e.g. 50 Hz de
 * interferencia de red eléctrica) mientras deja pasar el resto del espectro.
 * La transformación bilineal mapea el dominio analógico H(s) al digital H(z)
 * preservando la respuesta en frecuencia con warping mínimo cerca de f_notch.
 */

// ---------------------------------------------------------------------------
// Tipos públicos
// ---------------------------------------------------------------------------

/** Coeficientes del filtro biquad: numerador (b) y denominador (a) */
export interface BiquadCoefficients {
  /** b0: ganancia en paso de banda (numerador z^0) */
  b0: number;
  /** b1: coeficiente de cero en z^-1 (crea el null en f_notch) */
  b1: number;
  /** b2: coeficiente de cero en z^-2 (espejo de b0 para fase lineal) */
  b2: number;
  /** a1: coeficiente de polo en z^-1 (controla el ancho de la muesca) */
  a1: number;
  /** a2: coeficiente de polo en z^-2 (junto a a1 define la resonancia Q) */
  a2: number;
}

// ---------------------------------------------------------------------------
// Clase principal
// ---------------------------------------------------------------------------

export class NotchFilter {
  // ── Coeficientes (inmutables tras construcción) ──────────────────────────
  readonly coefficients: BiquadCoefficients;

  // ── Estado interno: registros de retardo (Direct Form II Transposed) ─────
  // w[0] = estado presente, w[1] = estado retrasado un sample
  private w0 = 0;
  private w1 = 0;

  /**
   * @param notchFreq  Frecuencia a eliminar en Hz        (default 50 Hz)
   * @param sampleRate Frecuencia de muestreo en Hz       (default 250 Hz)
   * @param Q          Factor de calidad (ancho de muesca) (default 30.0)
   *
   * Q alto → muesca muy estrecha → útil para EEG donde 49–51 Hz debe eliminarse
   * pero 40–48 Hz (ritmo gamma bajo) debe conservarse intacto.
   */
  constructor(
    public readonly notchFreq: number = 50,
    public readonly sampleRate: number = 250,
    public readonly Q: number = 30.0
  ) {
    this.coefficients = this._computeCoefficients(notchFreq, sampleRate, Q);
  }

  // ---------------------------------------------------------------------------
  // Cálculo de coeficientes (transformación bilineal para notch)
  // ---------------------------------------------------------------------------

  /**
   * Derivación matemática:
   *
   * 1. Frecuencia digital normalizada:
   *      ω₀ = 2π · f_notch / f_s
   *
   * 2. Pre-warping (corrección de distorsión bilineal en ω₀):
   *      ω₀_w = 2 · f_s · tan(ω₀ / 2)   [rad/s analógico]
   *      → En implementación discreta basta con trabajar con ω₀ directamente.
   *
   * 3. Ancho de banda a -3 dB:
   *      BW = ω₀ / Q   (en radianes/sample)
   *
   * 4. Función de transferencia del notch en el dominio z:
   *
   *      H(z) = (1 - 2·cos(ω₀)·z⁻¹ + z⁻²) / (1 - 2·r·cos(ω₀)·z⁻¹ + r²·z⁻²)
   *
   *    donde el radio del polo:
   *      r = 1 - π·BW / f_s  = 1 - (π · f_notch) / (Q · f_s)
   *
   *    Los ceros se sitúan exactamente sobre el círculo unitario en ±ω₀
   *    (atenuación infinita teórica en f_notch).
   *    Los polos se ubican ligeramente dentro del círculo (r < 1) para
   *    estabilidad, a la misma frecuencia angular, recuperando la ganancia
   *    unitaria fuera de la muesca.
   *
   * 5. Normalización por a0 (= 1 en esta forma):
   *    Los coeficientes ya están normalizados; a0 = 1 implícito.
   */
  private _computeCoefficients(
    f0: number,
    fs: number,
    Q: number
  ): BiquadCoefficients {
    // ω₀: frecuencia angular digital en radianes/sample
    const omega0 = (2 * Math.PI * f0) / fs;

    // cos(ω₀): determina la posición de ceros y polos sobre el eje angular
    const cosW0 = Math.cos(omega0);

    // Radio del polo r ∈ (0, 1):
    //   r → 1 : muesca cada vez más estrecha (Q alto)
    //   r → 0 : muesca muy ancha (filtro degenerado)
    const r = 1 - (Math.PI * f0) / (Q * fs);

    // ── Coeficientes del numerador (zeros en e^±jω₀) ─────────────────────
    // b0 = 1: ganancia unitaria en DC y Nyquist
    const b0 = 1;
    // b1 = -2·cos(ω₀): crea la cancelación exacta en f_notch
    const b1 = -2 * cosW0;
    // b2 = 1: simetría del numerador → respuesta de fase aproximadamente lineal
    const b2 = 1;

    // ── Coeficientes del denominador (polos en r·e^±jω₀) ─────────────────
    // a0 = 1 (implícito, no se almacena)
    // a1 = -2·r·cos(ω₀): desplaza los polos al mismo ángulo que los ceros
    const a1 = -2 * r * cosW0;
    // a2 = r²: módulo al cuadrado del polo; r < 1 garantiza estabilidad BIBO
    const a2 = r * r;

    // Normalización: dividir todo por a0 = 1 (ya normalizado)
    // En un biquad genérico se normalizaría por a0 si ≠ 1.

    return { b0, b1, b2, a1, a2 };
  }

  // ---------------------------------------------------------------------------
  // Procesamiento de muestras — Direct Form II Transposed
  // ---------------------------------------------------------------------------

  /**
   * Direct Form II Transposed minimiza operaciones de memoria y es numéricamente
   * más estable que Direct Form I para coeficientes de alta precisión.
   *
   * Ecuaciones de diferencia:
   *   y[n] = b0·x[n] + w0[n]
   *   w0[n+1] = b1·x[n] - a1·y[n] + w1[n]
   *   w1[n+1] = b2·x[n] - a2·y[n]
   *
   * @param sample Muestra de entrada x[n]
   * @returns      Muestra filtrada y[n]
   */
  process(sample: number): number {
    const { b0, b1, b2, a1, a2 } = this.coefficients;

    // Salida actual
    const output = b0 * sample + this.w0;

    // Actualización de estados (registros de retardo)
    const newW0 = b1 * sample - a1 * output + this.w1;
    const newW1 = b2 * sample - a2 * output;

    this.w0 = newW0;
    this.w1 = newW1;

    return output;
  }

  /**
   * Resetea los registros internos de estado.
   * Llamar antes de procesar una nueva sesión/epoch para evitar
   * transitorios causados por condiciones iniciales no nulas.
   */
  reset(): void {
    this.w0 = 0;
    this.w1 = 0;
  }
}

// ---------------------------------------------------------------------------
// Función de test
// ---------------------------------------------------------------------------

/**
 * Genera señales sinusoidales puras, las filtra y mide la atenuación en dB.
 *
 * Atenuación [dB] = 20 · log10(RMS_salida / RMS_entrada)
 * Valor negativo → atenuación; cerca de 0 → paso sin pérdida.
 */
export function testNotchFilter(): void {
  const SAMPLE_RATE = 250;   // Hz
  const NOTCH_FREQ  = 50;    // Hz — interferencia de red europea
  const Q           = 30.0;  // Factor de calidad estrecho para EEG
  const DURATION_S  = 2;     // segundos de señal de prueba
  const N           = SAMPLE_RATE * DURATION_S; // número de muestras

  const filter = new NotchFilter(NOTCH_FREQ, SAMPLE_RATE, Q);

  console.log("═══════════════════════════════════════════════════════");
  console.log(" Test NotchFilter — Biquad IIR Direct Form II Transposed");
  console.log("═══════════════════════════════════════════════════════");
  console.log(`  Notch freq : ${NOTCH_FREQ} Hz`);
  console.log(`  Sample rate: ${SAMPLE_RATE} Hz`);
  console.log(`  Q factor   : ${Q}`);
  console.log("\n  Coeficientes calculados:");
  const c = filter.coefficients;
  console.log(`    b0 = ${c.b0.toFixed(8)}  (ganancia numerador z^0)`);
  console.log(`    b1 = ${c.b1.toFixed(8)}  (cero en f_notch)`);
  console.log(`    b2 = ${c.b2.toFixed(8)}  (simetría numerador)`);
  console.log(`    a1 = ${c.a1.toFixed(8)}  (polo en f_notch, dentro |z|<1)`);
  console.log(`    a2 = ${c.a2.toFixed(8)}  (radio² del polo)`);
  console.log("");

  /**
   * Mide atenuación de una señal senoidal pura en dB.
   * Descarta los primeros 250 samples para ignorar el transitorio inicial.
   */
  function measureAttenuation(freqHz: number): number {
    filter.reset();

    const SKIP = SAMPLE_RATE; // descartar 1 s de transitorio
    let sumIn  = 0;
    let sumOut = 0;

    for (let n = 0; n < N; n++) {
      const x = Math.sin((2 * Math.PI * freqHz * n) / SAMPLE_RATE);
      const y = filter.process(x);

      if (n >= SKIP) {
        sumIn  += x * x;
        sumOut += y * y;
      }
    }

    const rmsIn  = Math.sqrt(sumIn  / (N - SKIP));
    const rmsOut = Math.sqrt(sumOut / (N - SKIP));

    return 20 * Math.log10(rmsOut / rmsIn); // dB (negativo = atenuación)
  }

  // ── Test 1: señal en la frecuencia de notch (debe atenuarse > 40 dB) ────
  const attn50 = measureAttenuation(50);
  const pass50 = attn50 < -40;
  console.log("  ┌─ TEST 1: Señal a 50 Hz (frecuencia notch)");
  console.log(`  │  Atenuación: ${attn50.toFixed(2)} dB`);
  console.log(`  │  Umbral requerido: < -40 dB`);
  console.log(`  └─ ${pass50 ? "✅ PASS" : "❌ FAIL"}`);
  console.log("");

  // ── Test 2: señal en banda EEG (10 Hz, debe pasar sin atenuar < 1 dB) ──
  const attn10 = measureAttenuation(10);
  const pass10 = attn10 > -1.0;
  console.log("  ┌─ TEST 2: Señal a 10 Hz (banda alfa EEG)");
  console.log(`  │  Atenuación: ${attn10.toFixed(4)} dB`);
  console.log(`  │  Umbral requerido: > -1 dB`);
  console.log(`  └─ ${pass10 ? "✅ PASS" : "❌ FAIL"}`);
  console.log("");

  // ── Test 3: señal a 60 Hz con filtro configurado para 60 Hz ─────────────
  const filter60 = new NotchFilter(60, SAMPLE_RATE, Q);
  let sumIn60 = 0, sumOut60 = 0;
  const SKIP = SAMPLE_RATE;
  for (let n = 0; n < N; n++) {
    const x = Math.sin((2 * Math.PI * 60 * n) / SAMPLE_RATE);
    const y = filter60.process(x);
    if (n >= SKIP) { sumIn60 += x * x; sumOut60 += y * y; }
  }
  const rmsIn60  = Math.sqrt(sumIn60  / (N - SKIP));
  const rmsOut60 = Math.sqrt(sumOut60 / (N - SKIP));
  const attn60   = 20 * Math.log10(rmsOut60 / rmsIn60);
  const pass60   = attn60 < -40;
  console.log("  ┌─ TEST 3: Filtro configurado a 60 Hz (red eléctrica USA)");
  console.log(`  │  Atenuación a 60 Hz: ${attn60.toFixed(2)} dB`);
  console.log(`  └─ ${pass60 ? "✅ PASS" : "❌ FAIL"}`);
  console.log("");

  const allPass = pass50 && pass10 && pass60;
  console.log("═══════════════════════════════════════════════════════");
  console.log(`  RESULTADO GLOBAL: ${allPass ? "✅ TODOS LOS TESTS PASARON" : "❌ ALGUNOS TESTS FALLARON"}`);
  console.log("═══════════════════════════════════════════════════════");
}

// ---------------------------------------------------------------------------
// Entry point (ejecutar directamente con ts-node o tras compilar)
// ---------------------------------------------------------------------------
testNotchFilter();
