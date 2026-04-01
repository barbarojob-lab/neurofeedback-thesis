/**
 * butterworth-filter.ts
 * Filtro Butterworth bandpass de orden 4, implementado como cascada de
 * 2 secciones biquad IIR (Second-Order Sections, SOS) en Direct Form II
 * Transposed.
 *
 * ── Justificación de diseño ────────────────────────────────────────────────
 *
 * Butterworth maximally-flat magnitude:
 *   - Respuesta plana (sin ripple) en la banda de paso → no distorsiona la
 *     amplitud relativa de ondas delta/theta/alpha/beta EEG.
 *   - Rolloff de 80 dB/década para orden 4 → suficiente para rechazar DC
 *     (drift de electrodo) por debajo de 1 Hz y artefacto EMG por encima
 *     de 30 Hz.
 *
 * Por qué highCut = 30 Hz para EEG clínico:
 *   - Beta de interés: 12–30 Hz (cognición, motor imagery).
 *   - Artefacto EMG muscular facial: componentes desde ~20 Hz hasta >200 Hz,
 *     con mayor potencia sobre electrodos frontales (Fp1, Fp2, Fz).
 *   - Corte en 30 Hz elimina ~90 % de potencia EMG sin requerir ICA,
 *     preservando latencia de respuesta evocada < 100 ms.
 *   - Referencia: Thibault et al., NeuroImage 2018, "Misconceptions about
 *     EEG artefact rejection."
 *
 * Cascada de 2 biquads vs. filtro directo orden 4:
 *   - Un polinomio de orden 4 tiene coeficientes grandes → sensibilidad
 *     numérica alta (efecto "butterfly" en punto flotante de 64 bits).
 *   - Factorización en SOS mantiene coeficientes O(1) → estabilidad
 *     garantizada por inspección de cada sección individual.
 *
 * ── Método de diseño ──────────────────────────────────────────────────────
 *
 * 1. Prototipo lowpass Butterworth analógico de orden N=2 (polos en el
 *    semicírculo unitario izquierdo del plano-s).
 *
 * 2. Transformación lowpass→bandpass analógica: mapea 1 polo LP en 2 polos BP
 *    por cada polo original → orden efectivo 2N = 4, con 2 secciones SOS.
 *
 * 3. Pre-warping de frecuencias de corte para corregir compresión bilineal:
 *      Ω_c = 2·f_s · tan(π·f_c / f_s)   [rad/s analógico pre-warped]
 *
 * 4. Transformación bilineal s→z por sección:
 *      s = 2·f_s · (z-1)/(z+1)
 *    aplicada analíticamente a cada factor cuadrático del denominador BP.
 */

// ---------------------------------------------------------------------------
// Tipos públicos
// ---------------------------------------------------------------------------

/**
 * Coeficientes de una sección biquad (Second-Order Section).
 * Función de transferencia:
 *
 *   H_k(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)
 *
 * Nota: a0 = 1 implícito (normalizado).
 */
export interface BiquadSection {
  /** Índice de la sección (0 o 1 para orden 4) */
  index: number;
  /** b0: ganancia de la sección en DC relativo */
  b0: number;
  /** b1: coeficiente z⁻¹ del numerador */
  b1: number;
  /** b2: coeficiente z⁻² del numerador */
  b2: number;
  /** a1: coeficiente z⁻¹ del denominador (normalizado, a0=1) */
  a1: number;
  /** a2: coeficiente z⁻² del denominador */
  a2: number;
}

// ---------------------------------------------------------------------------
// Estado interno de una sección biquad
// ---------------------------------------------------------------------------

/** Registros de retardo para Direct Form II Transposed */
interface BiquadState {
  w0: number; // estado presente
  w1: number; // estado retrasado un sample
}

// ---------------------------------------------------------------------------
// Clase principal
// ---------------------------------------------------------------------------

export class ButterworthBandpass {
  /** Las 2 secciones SOS calculadas en el constructor */
  private readonly sections: BiquadSection[];

  /** Estados internos de cada sección (2 registros por sección) */
  private states: BiquadState[];

  /**
   * @param lowCut     Frecuencia de corte inferior [Hz]  (default 1 Hz)
   * @param highCut    Frecuencia de corte superior [Hz]  (default 30 Hz)
   * @param sampleRate Frecuencia de muestreo [Hz]        (default 250 Hz)
   *
   * ⚠️  highCut = 30 Hz es el valor clínico recomendado para EEG (ver cabecera).
   *     Modificar solo si el protocolo experimental lo justifica explícitamente.
   */
  constructor(
    public readonly lowCut: number = 1,
    public readonly highCut: number = 30,
    public readonly sampleRate: number = 250
  ) {
    if (lowCut <= 0)              throw new Error("lowCut debe ser > 0 Hz");
    if (highCut <= lowCut)        throw new Error("highCut debe ser > lowCut");
    if (highCut >= sampleRate / 2) throw new Error("highCut debe ser < Nyquist");

    this.sections = this._designButterworthBP(lowCut, highCut, sampleRate);
    this.states   = this.sections.map(() => ({ w0: 0, w1: 0 }));
  }

  // ---------------------------------------------------------------------------
  // Diseño del filtro — transformación bilineal con pre-warping
  // ---------------------------------------------------------------------------

  /**
   * Diseña 2 secciones SOS Butterworth bandpass orden 4.
   *
   * ── Paso 1: Pre-warping ──────────────────────────────────────────────────
   *
   *   La transformación bilineal comprime el eje de frecuencias real (0…∞)
   *   al eje digital (0…π) con una distorsión no lineal (warping). Las
   *   frecuencias de corte deben pre-warpearse para que la banda de paso
   *   digital sea exactamente [lowCut, highCut] Hz.
   *
   *     ω_L = 2·f_s · tan(π·f_L / f_s)   [rad/s analógico]
   *     ω_H = 2·f_s · tan(π·f_H / f_s)
   *
   * ── Paso 2: Parámetros del prototipo bandpass ────────────────────────────
   *
   *     BW  = ω_H - ω_L        [ancho de banda analógico]
   *     ω₀² = ω_L · ω_H        [frecuencia central geométrica al cuadrado]
   *
   * ── Paso 3: Polos del prototipo LP Butterworth orden 2 ──────────────────
   *
   *   Para N=2, los 2 polos del prototipo LP unitario se ubican en:
   *     p_k = e^{j·π·(2k+N-1)/(2N)}  para k = 0, 1
   *
   *   Con N=2:
   *     p₀ = e^{j·3π/4} = -√2/2 + j·√2/2
   *     p₁ = e^{j·5π/4} = -√2/2 - j·√2/2
   *
   *   (conjugados: solo necesitamos p₀, su conjugado da la misma sección)
   *
   * ── Paso 4: Transformación LP→BP analógica ───────────────────────────────
   *
   *   Sustitución: s_LP → (s² + ω₀²) / (BW·s)
   *
   *   Cada polo LP p_k genera 2 polos BP resolviendo:
   *     s² - BW·p_k·s + ω₀² = 0
   *     s = (BW·p_k ± √(BW²·p_k² - 4·ω₀²)) / 2
   *
   *   Con p₀ complejo, los 4 raíces son 2 pares conjugados → 2 secciones
   *   de segundo orden reales.
   *
   * ── Paso 5: Transformación bilineal s→z por sección ─────────────────────
   *
   *   Para un denominador analógico (s - s_a)(s - conj(s_a)) = s² + α·s + β:
   *
   *   Aplicando s = 2·f_s·(1-z⁻¹)/(1+z⁻¹):
   *
   *   Denominador digital (normalizado por a0):
   *     a0 = 4·f_s² + 2·α·f_s + β  (factor de normalización)
   *     a1 = (2β - 8·f_s²) / a0
   *     a2 = (4·f_s² - 2·α·f_s + β) / a0
   *
   *   Numerador de cada sección BP: la transformación LP→BP produce un
   *   numerador s en el analógico → (1 - z⁻²) en digital (bandpass puro):
   *     b_raw = [BW/f_s², 0, -BW/f_s²] antes de normalizar
   *     b0 = BW·2·f_s / a0
   *     b1 = 0
   *     b2 = -b0
   */
  private _designButterworthBP(
    fL: number,
    fH: number,
    fs: number
  ): BiquadSection[] {
    // ── Pre-warping de frecuencias de corte ──────────────────────────────
    // Corrección de la distorsión no lineal de la transformación bilineal.
    // tan(π·f/fs) aproxima bien para f << fs/2; con fs=250, fL=1: tan≈0.0126
    const wL = 2 * fs * Math.tan((Math.PI * fL) / fs); // rad/s analógico pre-warped bajo
    const wH = 2 * fs * Math.tan((Math.PI * fH) / fs); // rad/s analógico pre-warped alto

    // ── Parámetros del prototipo bandpass analógico ──────────────────────
    // BW: ancho de banda de -3 dB en rad/s
    const BW  = wH - wL;
    // w0sq: frecuencia central geométrica al cuadrado (ω₀² = ωL·ωH)
    //   Garantiza simetría logarítmica de la respuesta en frecuencia.
    const w0sq = wL * wH;

    // ── Polos del prototipo Butterworth LP de orden 2 ────────────────────
    // Para N=2, ángulos θ_k = π(2k+N-1)/(2N), k=0,1
    // Solo necesitamos k=0 (par conjugado genera 1 sección real de orden 2):
    //   θ₀ = 3π/4 → polo: -√2/2 + j·√2/2
    //
    // Parte real e imaginaria del polo LP prototipo:
    const lpPoleRe = -Math.SQRT2 / 2; // = cos(3π/4)
    const lpPoleIm =  Math.SQRT2 / 2; // = sin(3π/4)

    // ── Transformación LP→BP para cada polo LP ───────────────────────────
    // Cada polo complejo p_k = re + j·im genera 2 secciones BP.
    // Resolvemos s² - BW·p_k·s + w0² = 0 para k=0 y k=1 (conj. de k=0):
    //
    // k=0: p₀ = lpPoleRe + j·lpPoleIm
    // k=1: p₁ = lpPoleRe - j·lpPoleIm  (conjugado)
    //
    // Las raíces de cada ecuación cuadrática son pares conjugados entre sí,
    // por lo que cada par genera exactamente 1 sección biquad real.

    const sections: BiquadSection[] = [];

    for (const sign of [+1, -1] as const) {
      // Polo LP: p = lpPoleRe ± j·lpPoleIm
      const pRe = lpPoleRe;
      const pIm = sign * lpPoleIm;

      // Coeficientes del cuadrático analógico BP:
      // s² - BW·p·s + w0² = 0
      // Parte real del coeficiente lineal: -BW·pRe
      // Parte imaginaria: -BW·pIm  (debe ser cero para sección real)
      //
      // Discriminante: Δ = (BW·p)² - 4·w0²
      //   Δ_re = BW²·(pRe² - pIm²) - 4·w0²
      //   Δ_im = BW²·2·pRe·pIm
      const BWpRe = BW * pRe; // parte real de BW·p
      const BWpIm = BW * pIm; // parte imaginaria de BW·p

      const discRe = BWpRe * BWpRe - BWpIm * BWpIm - 4 * w0sq;
      const discIm = 2 * BWpRe * BWpIm;

      // Raíz cuadrada compleja del discriminante: √(discRe + j·discIm)
      const discMod = Math.sqrt(discRe * discRe + discIm * discIm);
      const discArg = Math.atan2(discIm, discRe);
      const sqrtRe  = Math.sqrt(discMod) * Math.cos(discArg / 2);
      const sqrtIm  = Math.sqrt(discMod) * Math.sin(discArg / 2);

      // Raíces del cuadrático BP (2 raíces conjugadas → sección real):
      //   s₊ = (BW·p + √Δ) / 2
      //   s₋ = (BW·p - √Δ) / 2
      const s1Re = (BWpRe + sqrtRe) / 2;
      const s1Im = (BWpIm + sqrtIm) / 2;

      // La sección real de segundo orden tiene denominador:
      //   (s - s1)(s - conj(s1)) = s² - 2·s1Re·s + (s1Re² + s1Im²)
      //
      // α = -2·s1Re   (coeficiente lineal real del denominador analógico)
      // β = s1Re² + s1Im²  (término independiente = módulo² del polo)
      const alpha = -2 * s1Re; // α > 0 si s1Re < 0 (polo en semiplano izq → estable)
      const beta  = s1Re * s1Re + s1Im * s1Im;

      // ── Transformación bilineal s→z ────────────────────────────────────
      // Denominador analógico: s² + α·s + β
      // Con s = 2fs·(1-z⁻¹)/(1+z⁻¹) = K·(1-z⁻¹)/(1+z⁻¹), K = 2fs:
      //
      //   Denominador digital (antes de normalizar):
      //   = K²(1-z⁻¹)² + α·K·(1-z⁻¹)(1+z⁻¹) + β·(1+z⁻¹)²
      //
      //   Coeficiente z⁰ (a0): K² + α·K + β
      //   Coeficiente z⁻¹(a1): -2K² + 2β          (normalizar por a0)
      //   Coeficiente z⁻²(a2): K² - α·K + β       (normalizar por a0)

      const K   = 2 * fs; // factor bilineal (= 2·fs con pre-warping ya incorporado)
      const K2  = K * K;

      const a0_raw = K2 + alpha * K + beta;  // factor de normalización
      const a1_raw = -2 * K2 + 2 * beta;     // coef. z⁻¹ antes de norm.
      const a2_raw =  K2 - alpha * K + beta; // coef. z⁻² antes de norm.

      // ── Numerador del bandpass en digital ─────────────────────────────
      // El prototipo LP→BP produce un factor 's' en el numerador analógico.
      // Con la transformación bilineal, 's' mapea a:
      //   K·(1-z⁻¹)/(1+z⁻¹)  →  numerador ∝ (1 - z⁻²) tras multiplicar por (1+z⁻¹)
      //
      // El numerador digital es proporcional a (1 - z⁻²):
      //   Ganancia: BW·K / a0_raw  (BW del BP analógico, K de la bilineal)
      //
      // Esto da ganancia unitaria en la frecuencia central ω₀.

const gain = BW * K / a0_raw;

const b0 =  gain; // coeficiente z⁰  del numerador
const b1 =  0;    // coeficiente z⁻¹ (cero exacto → simetría del BP)
const b2 = -gain; // coeficiente z⁻² (= -b0, antisimétrico)


      // Denominador normalizado (dividido por a0_raw):
      const a1 = a1_raw / a0_raw;
      const a2 = a2_raw / a0_raw;

      sections.push({ index: sections.length, b0, b1, b2, a1, a2 });
    }

    return sections; // 2 secciones → orden total 4
  }

  // ---------------------------------------------------------------------------
  // Procesamiento de muestras — cascada de secciones
  // ---------------------------------------------------------------------------

  /**
   * Procesa una muestra pasándola secuencialmente por las 2 secciones SOS.
   *
   * Cascada: x[n] → [Sección 0] → y0[n] → [Sección 1] → y[n]
   *
   * Ecuaciones de diferencia por sección k (Direct Form II Transposed):
   *   y_k[n]   = b0_k·x_k[n] + w0_k[n]
   *   w0_k[n+1]= b1_k·x_k[n] - a1_k·y_k[n] + w1_k[n]
   *   w1_k[n+1]= b2_k·x_k[n] - a2_k·y_k[n]
   *
   * @param sample Muestra EEG cruda x[n] en µV
   * @returns      Muestra filtrada y[n] en µV
   */
  process(sample: number): number {
    let x = sample;

    for (let k = 0; k < this.sections.length; k++) {
      const section = this.sections[k]!;
      const { b0, b1, b2, a1, a2 } = section;
      const st = this.states[k]!;

      // Salida de la sección k
      const y = b0 * x + st.w0;

      // Actualización de registros de retardo
      const newW0 = b1 * x - a1 * y + st.w1;
      const newW1 = b2 * x - a2 * y;

      st.w0 = newW0;
      st.w1 = newW1;

      // La salida de esta sección es la entrada de la siguiente
      x = y;
    }

    return x;
  }

  /**
   * Resetea todos los estados internos a cero.
   * Llamar al inicio de cada epoch o sesión para evitar transitorios.
   */
  reset(): void {
    for (let i = 0; i < this.states.length; i++) {
      const st = this.states[i]!;
      st.w0 = 0;
      st.w1 = 0;
    }
  }

  /**
   * Retorna una copia de los coeficientes de las 2 secciones SOS.
   * Útil para logging, serialización o verificación externa de coeficientes.
   */
  getSections(): BiquadSection[] {
    return this.sections.map(s => ({ ...s }));
  }
}

// ---------------------------------------------------------------------------
// Función de test
// ---------------------------------------------------------------------------

/**
 * Mide la atenuación RMS de una señal sinusoidal pura tras N ciclos de
 * estado estacionario (descartando el transitorio inicial).
 *
 * @param filter  Instancia de ButterworthBandpass ya construida
 * @param freqHz  Frecuencia de la señal de prueba en Hz
 * @param fs      Frecuencia de muestreo en Hz
 * @returns       Atenuación en dB (negativo = atenuación, 0 = paso libre)
 */
function measureAttenuation(
  filter: ButterworthBandpass,
  freqHz: number,
  fs: number
): number {
  filter.reset();

  const TOTAL_S  = 4;        // segundos totales de señal
  const SKIP_S   = 2;        // segundos a descartar (transitorio + estado estacionario)
  const N        = fs * TOTAL_S;
  const SKIP     = fs * SKIP_S;

  let sumIn  = 0;
  let sumOut = 0;

  for (let n = 0; n < N; n++) {
    const x = Math.sin((2 * Math.PI * freqHz * n) / fs);
    const y = filter.process(x);

    if (n >= SKIP) {
      sumIn  += x * x;
      sumOut += y * y;
    }
  }

  const rmsIn  = Math.sqrt(sumIn  / (N - SKIP));
  const rmsOut = Math.sqrt(sumOut / (N - SKIP));

  if (rmsIn < 1e-12) return -Infinity;
  return 20 * Math.log10(rmsOut / rmsIn);
}

export function testButterworthFilter(): void {
  const FS       = 250;  // Hz
  const LOW_CUT  = 1;    // Hz
  const HIGH_CUT = 30;   // Hz (ver justificación en cabecera)

  const filter = new ButterworthBandpass(LOW_CUT, HIGH_CUT, FS);

  console.log("═══════════════════════════════════════════════════════════════");
  console.log(" Test ButterworthBandpass — Orden 4, SOS (2 × biquad IIR)");
  console.log("═══════════════════════════════════════════════════════════════");
  console.log(`  Banda de paso : ${LOW_CUT} – ${HIGH_CUT} Hz`);
  console.log(`  Sample rate   : ${FS} Hz`);
  console.log(`  Secciones SOS : ${filter.getSections().length}`);
  console.log("");

  for (const sec of filter.getSections()) {
    console.log(`  ── Sección ${sec.index} ──────────────────────────────────────`);
    console.log(`    b0 = ${sec.b0.toFixed(10)}   (ganancia; nota: b1=0, b2=-b0)`);
    console.log(`    a1 = ${sec.a1.toFixed(10)}   (polo z⁻¹)`);
    console.log(`    a2 = ${sec.a2.toFixed(10)}   (polo z⁻²)`);
  }
  console.log("");

  // Definición de casos de prueba
  const tests: Array<{
    label: string;
    freqHz: number;
    thresholdDb: number;
    condition: "below" | "above"; // "below" = debe atenuarse, "above" = debe pasar
  }> = [
    {
      label: "0.5 Hz (drift DC, fuera de banda inferior)",
      freqHz: 0.5,
      thresholdDb: -20,
      condition: "below",
    },
    {

      label: "6 Hz (Theta, dentro de banda de paso)",
      freqHz: 6,
      thresholdDb: -1,
      condition: "above",

    },
    {
      label: "50 Hz (interferencia red, fuera de banda superior)",
      freqHz: 50,
      thresholdDb: -20,
      condition: "below",
    },
  ];

  let allPass = true;

  for (const t of tests) {
    const attn = measureAttenuation(filter, t.freqHz, FS);
    const pass =
      t.condition === "below" ? attn < t.thresholdDb : attn > t.thresholdDb;

    allPass = allPass && pass;

    const condStr =
      t.condition === "below"
        ? `< ${t.thresholdDb} dB (debe atenuarse)`
        : `> ${t.thresholdDb} dB (debe pasar)`;

    console.log(`  ┌─ TEST: ${t.label}`);
    console.log(`  │  Atenuación medida : ${attn.toFixed(3)} dB`);
    console.log(`  │  Condición         : ${condStr}`);
    console.log(`  └─ ${pass ? "✅ PASS" : "❌ FAIL"}`);
    console.log("");
  }

  console.log("═══════════════════════════════════════════════════════════════");
  console.log(
    `  RESULTADO GLOBAL: ${allPass ? "✅ TODOS LOS TESTS PASARON" : "❌ ALGUNOS TESTS FALLARON"}`
  );
  console.log("═══════════════════════════════════════════════════════════════");
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
testButterworthFilter();
