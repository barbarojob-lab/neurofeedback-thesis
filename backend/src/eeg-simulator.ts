/**
 * @file src/eeg-simulator.ts
 * @description Simulador de señal EEG sintética realista para desarrollo y testing
 *              del pipeline de neurofeedback sin hardware físico.
 *
 * ── Modelo de señal ──────────────────────────────────────────────────────────
 *
 *  EEG_sintetico(t) = Σ [Ai · sin(2π · fi · t + φi)]  +  η(t)  +  artefacto(t)
 *
 *  Donde:
 *    Ai   = amplitud de la banda i (μV)
 *    fi   = frecuencia central de la banda i (Hz)
 *    φi   = fase inicial aleatoria (0–2π) — cada canal tiene fases distintas
 *    η(t) = ruido gaussiano blanco (σ = 5 μV) — fondo de actividad cortical
 *    artefacto(t) = interferencia de red eléctrica a 50 Hz (15 μV)
 *
 *  Esta superposición de senoidales es la aproximación de primer orden
 *  más usada en simuladores EEG académicos (p. ej. EEGsynth, MNE-Python
 *  simulate_raw). No modela correlaciones espaciales entre canales ni
 *  dinámica no-estacionaria, pero es suficiente para testear el pipeline
 *  de filtrado → FFT → TBR → z-score → feedback.
 *
 * ── Uso básico ───────────────────────────────────────────────────────────────
 *
 *  const sim = new EEGSimulator({ sampleRate: 250, channels: 8 });
 *  sim.on('sample', (s: EEGSample) => pipeline.process(s));
 *  sim.start();
 *
 *  // Simular aumento manual de theta (para testear thresholds):
 *  sim.setTheta(60);
 *
 *  // Simular rampa de trance (para testear la respuesta del feedback E2E):
 *  sim.enableTranceMode();
 *
 *  sim.stop();
 */

// Self-contained typed EventEmitter (no @types/node dependency for compilation)
type Listener = (...args: unknown[]) => void;

class EventEmitter {
  private _listeners: Map<string, Listener[]> = new Map();
  on(event: string, listener: Listener): this {
    const arr = this._listeners.get(event) ?? [];
    arr.push(listener);
    this._listeners.set(event, arr);
    return this;
  }
  off(event: string, listener: Listener): this {
    const arr = (this._listeners.get(event) ?? []).filter(l => l !== listener);
    this._listeners.set(event, arr);
    return this;
  }
  emit(event: string, ...args: unknown[]): boolean {
    const arr = this._listeners.get(event) ?? [];
    arr.forEach(l => l(...args));
    return arr.length > 0;
  }
}
import type { EEGSample, SessionConfig } from './types';

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTES FISIOLÓGICAS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Frecuencias centrales representativas de cada banda.
 *
 * Se elige UNA frecuencia por banda en lugar de ruido de banda ancha porque:
 *   1. Permite control preciso de amplitud en el simulador.
 *   2. La FFT del pipeline la detecta limpiamente → facilita validación unitaria.
 *   3. Es la convención en simuladores EEG académicos (Klem et al., 1999).
 */
const FREQ = {
  DELTA: 2,   // Hz — onda representativa del ritmo delta (0.5–4 Hz)
  THETA: 6,   // Hz — centro del ritmo theta (4–8 Hz), frecuencia más activa en hipnosis
  ALPHA: 10,  // Hz — pico del ritmo alfa (8–12 Hz), dominante con ojos cerrados
  BETA:  20,  // Hz — beta medio (13–30 Hz), alerta cognitiva moderada
  LINE:  50,  // Hz — interferencia de red eléctrica europea (60 Hz en EEUU/Japón)
} as const;

/**
 * Amplitudes base en microvoltios (μV).
 *
 *   Delta (80 μV): banda más energética — la mayor amplitud es fisiológicamente
 *     correcta: delta tiene la mayor potencia espectral en EEG de reposo.
 *
 *   Theta (30 μV): amplitud típica en adultos en reposo con ojos cerrados.
 *     Aumenta a 60–100 μV en meditación profunda o trance (Gruzelier, 2014).
 *     El modo trance simula esta subida de 30 → 80 μV.
 *
 *   Alpha (20 μV): amplitud media del ritmo alfa occipital con ojos cerrados.
 *     En el simulador se mantiene constante (sesión siempre con ojos cerrados).
 *
 *   Beta (10 μV): amplitud típica de beta frontocentral en reposo.
 *     Baja en trance. El protocolo busca TBR > 1, es decir theta > beta.
 *
 *   Ruido (5 μV): SD del ruido de fondo en EEG de buena calidad (<5 kΩ impedancia).
 *     En mala calidad puede llegar a 50 μV; 5 μV representa sesión bien preparada.
 *
 *   Artefacto de red (15 μV): interferencia de 50 Hz. El filtro notch debe
 *     atenuarla >40 dB (a <1 μV). Amplitud moderada, realista para lab sin
 *     jaula de Faraday.
 */
const AMP_BASE = {
  DELTA: 80,
  THETA: 30,
  ALPHA: 20,
  BETA:  10,
  NOISE: 5,
  LINE:  15,
} as const;

/**
 * Parámetros de la rampa de modo trance.
 *
 * 30 s refleja la dinámica real de inducción hipnótica: los estudios de EEG
 * muestran que theta aumenta progresivamente en los primeros 2–5 minutos
 * (Terhune et al., 2011). Se usa 30 s en el simulador para pruebas E2E manejables.
 */
const TRANCE_MODE = {
  THETA_START:  30,      // μV — amplitud theta basal
  THETA_END:    80,      // μV — amplitud theta en trance profundo
  DURATION_MS:  30_000,  // ms — duración total de la rampa
} as const;

/** Número de bandas oscilatorias simuladas por canal. */
const NUM_BANDS = 5; // delta, theta, alpha, beta, line

// ─────────────────────────────────────────────────────────────────────────────
// TIPOS INTERNOS
// ─────────────────────────────────────────────────────────────────────────────

/** Opciones de configuración del simulador. */
export interface SimulatorOptions {
  /** Frecuencia de muestreo en Hz. Default: 250. */
  sampleRate?: 250 | 500;
  /** Número de canales EEG a simular. Default: 8. */
  channels?: number;
}

/** Estado interno mutable del simulador. */
interface SimulatorState {
  /** Tiempo transcurrido en segundos desde el inicio. Avanza discretamente. */
  t: number;
  /** Amplitud actual de theta (μV); modificable en tiempo real vía setTheta(). */
  thetaAmplitude: number;
  /**
   * Fases iniciales aleatorias por canal × banda.
   * Longitud: channels × NUM_BANDS.
   * Garantizan señales decorreladas entre canales, como en EEG real
   * (cada electrodo ve una mezcla distinta de fuentes dipoleares).
   */
  phases: Float32Array;
  /** ¿Está activa la rampa de modo trance? */
  tranceModeActive: boolean;
  /** Timestamp (ms) del inicio del modo trance para calcular el progreso de rampa. */
  tranceModeStart: number;
  /** Handle del setInterval; null cuando el simulador está parado. */
  intervalHandle: ReturnType<typeof setInterval> | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILIDADES MATEMÁTICAS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Genera un número aleatorio con distribución normal N(0, σ)
 * mediante la transformación de Box-Muller.
 *
 * Modela el ruido de fondo del EEG: actividad cortical asincrónica de
 * millones de neuronas que aparece como ruido blanco en el espectro.
 * Box-Muller es suficiente aquí; no se requiere Ziggurat.
 */
function gaussianNoise(sigma: number): number {
  const u1 = Math.random() || Number.EPSILON; // evitar log(0)
  const u2 = Math.random() || Number.EPSILON;
  return sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Interpolación lineal entre a y b, t ∈ [0, 1].
 * t se clampea a [0, 1] para evitar extrapolación fuera del rango de la rampa.
 */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * Math.max(0, Math.min(1, t));
}

// ─────────────────────────────────────────────────────────────────────────────
// CLASE PRINCIPAL
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Simulador de señal EEG sintética realista.
 *
 * Extiende EventEmitter para emitir 'sample' con cada EEGSample generado,
 * siendo drop-in replacement del driver de hardware en dev/test
 * (mismo interfaz de eventos que MuseDriver, OpenBCIDriver…).
 *
 * @emits 'sample'  - EEGSample completo en cada tick del setInterval
 * @emits 'started' - Al llamar a start() con éxito
 * @emits 'stopped' - Al llamar a stop() con éxito
 * @emits 'trance'  - Al activar el modo trance; payload: { targetTheta: number }
 */
export class EEGSimulator extends EventEmitter {

  // ── Configuración inmutable ────────────────────────────────────────────────

  readonly sampleRate: 250 | 500;
  readonly channels: number;

  /**
   * Intervalo de muestreo en ms = 1000 / sampleRate.
   * A 250 Hz → 4 ms/muestra.
   *
   * ⚠ Limitación: setInterval en Node.js no es tiempo real (jitter ±1–2 ms).
   * Para producción usar nanotimer (addon nativo) o timestamps del driver hardware.
   * Para simulación de señal y testing del pipeline, 4 ms es suficientemente preciso.
   */
  private readonly intervalMs: number;

  // ── Estado interno ─────────────────────────────────────────────────────────

  private state: SimulatorState;

  // ─────────────────────────────────────────────────────────────────────────────

  constructor(options: SimulatorOptions = {}) {
    super();

    this.sampleRate  = options.sampleRate ?? 250;
    this.channels    = options.channels   ?? 8;
    this.intervalMs  = 1000 / this.sampleRate;

    // Fase inicial aleatoria por canal × banda.
    // Sin fases distintas, todos los canales serían copias idénticas,
    // lo que no ocurre en EEG real (cada electrodo ve una combinación
    // diferente de las mismas fuentes corticales).
    const phases = new Float32Array(this.channels * NUM_BANDS);
    for (let i = 0; i < phases.length; i++) {
      phases[i] = Math.random() * 2 * Math.PI;
    }

    this.state = {
      t:                0,
      thetaAmplitude:   AMP_BASE.THETA,
      phases,
      tranceModeActive: false,
      tranceModeStart:  0,
      intervalHandle:   null,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // API PÚBLICA
  // ─────────────────────────────────────────────────────────────────────────────

  /**
   * Inicia la generación de muestras EEG sintéticas.
   * Emite el evento 'sample' (EEGSample) en cada tick del setInterval.
   * Lanza error si ya está corriendo para evitar intervalos duplicados.
   */
  start(): void {
    if (this.state.intervalHandle !== null) {
      throw new Error('[EEGSimulator] Ya está corriendo. Llama stop() primero.');
    }

    this.state.t = 0;

    this.state.intervalHandle = setInterval(() => {
      const sample = this.generateSample();
      this.emit('sample', sample);

      // Avanzar tiempo discretamente en lugar de usar Date.now() como variable
      // de fase, para que la señal sea determinista (no acumula jitter del timer).
      this.state.t += this.intervalMs / 1000; // ms → s
    }, this.intervalMs);

    this.emit('started');
    console.log(
      `[EEGSimulator] Iniciado — ${this.sampleRate} Hz | ` +
      `${this.channels} canales | intervalo ${this.intervalMs} ms`
    );
  }

  /**
   * Detiene la generación de muestras.
   * Seguro llamar aunque no esté corriendo (no lanza error).
   */
  stop(): void {
    if (this.state.intervalHandle !== null) {
      clearInterval(this.state.intervalHandle);
      this.state.intervalHandle = null;
    }
    this.state.tranceModeActive = false;
    this.emit('stopped');
    console.log('[EEGSimulator] Detenido.');
  }

  /**
   * Establece la amplitud de theta en tiempo real (μV).
   *
   * Permite inyectar manualmente distintos niveles de theta para verificar
   * que el motor de decisión genera los FeedbackCommand correctos:
   *
   *   sim.setTheta(80);  → TBR alto → 'increase_theta' reward
   *   sim.setTheta(10);  → TBR bajo → 'decrease_theta' alerta
   *   sim.setTheta(30);  → baseline → 'neutral'
   *
   * Desactiva el modo trance si está activo (evita conflicto de control).
   *
   * @param amplitude - Amplitud en μV. Rango fisiológico: 5–150 μV.
   */
  setTheta(amplitude: number): void {
    if (amplitude < 0) {
      throw new RangeError('[EEGSimulator] La amplitud no puede ser negativa.');
    }
    this.state.tranceModeActive = false; // evitar conflicto con la rampa
    this.state.thetaAmplitude   = amplitude;
    console.log(`[EEGSimulator] Theta → ${amplitude} μV`);
  }

  /**
   * Activa el modo trance: rampa lineal de theta 30 → 80 μV en 30 segundos.
   *
   * Propósito de test E2E:
   *   Con la rampa activa, el TBR sube progresivamente y el sistema de
   *   feedback debe responder emitiendo 'increase_theta' de forma sostenida.
   *   Permite verificar la latencia de respuesta del pipeline completo:
   *     EEGSample → BandPowers → TBR → ZScore → FeedbackCommand → WSMessage
   *
   * La dinámica de 30 s refleja el aumento real de theta en los primeros
   * minutos de inducción hipnótica (Terhune et al., 2011). Al completarse,
   * la amplitud queda fija en 80 μV (el simulador no vuelve solo al basal).
   */
  enableTranceMode(): void {
    this.state.tranceModeActive = true;
    this.state.tranceModeStart  = Date.now();
    this.state.thetaAmplitude   = TRANCE_MODE.THETA_START;
    this.emit('trance', { targetTheta: TRANCE_MODE.THETA_END });
    console.log(
      `[EEGSimulator] Modo trance activado: ` +
      `theta ${TRANCE_MODE.THETA_START} → ${TRANCE_MODE.THETA_END} μV ` +
      `en ${TRANCE_MODE.DURATION_MS / 1000} s`
    );
  }

  /** Desactiva el modo trance y restaura la amplitud theta basal (30 μV). */
  disableTranceMode(): void {
    this.state.tranceModeActive = false;
    this.state.thetaAmplitude   = AMP_BASE.THETA;
    console.log('[EEGSimulator] Modo trance desactivado. Theta → 30 μV.');
  }

  /** true si el simulador está generando muestras activamente. */
  get isRunning(): boolean {
    return this.state.intervalHandle !== null;
  }

  /** Amplitud theta actual en μV (refleja setTheta() o la rampa de trance). */
  get currentThetaAmplitude(): number {
    return this.state.thetaAmplitude;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // GENERACIÓN DE SEÑAL (métodos privados)
  // ─────────────────────────────────────────────────────────────────────────────

  /**
   * Genera un EEGSample completo para el instante t actual.
   * Actualiza la rampa de trance antes de calcular los canales.
   */
  private generateSample(): EEGSample {
    this.updateTranceRamp();

    const channelData = new Float32Array(this.channels);
    for (let ch = 0; ch < this.channels; ch++) {
      channelData[ch] = this.generateChannelSample(ch);
    }

    return {
      timestamp:  Date.now(),
      channelData,
      sampleRate: this.sampleRate,
    };
  }

  /**
   * Calcula el voltaje (μV) para un canal específico en el instante t.
   *
   * Modelo:
   *   V(t) = delta(t) + theta(t) + alpha(t) + beta(t) + line(t) + η(t)
   *
   * Cada componente es:  Ai · sin(2π · fi · t + φ_canal_banda)
   *
   * Las fases φ_canal_banda son distintas por canal para simular que
   * los electrodos están en posiciones diferentes respecto a las fuentes
   * corticales (modelo simplificado sin lead-field).
   */
  private generateChannelSample(ch: number): number {
    const t = this.state.t;
    const base = ch * NUM_BANDS; // índice base en el array de fases

    // ── Delta (2 Hz, 80 μV) ───────────────────────────────────────────────
    // La mayor amplitud es fisiológicamente correcta: delta domina el espectro
    // de potencia en el EEG de adulto en reposo (ley de 1/f del EEG).
    const delta = AMP_BASE.DELTA *
      Math.sin(2 * Math.PI * FREQ.DELTA * t + (this.state.phases[base + 0] ?? 0));

    // ── Theta (6 Hz, amplitud controlable) ───────────────────────────────
    // Biomarcador objetivo. 6 Hz: frecuencia más frecuentemente reportada
    // en estudios de hipnosis frontal y meditación (Bazanova & Vernon, 2014).
    // La amplitud es mutable (setTheta / modo trance).
    const theta = this.state.thetaAmplitude *
      Math.sin(2 * Math.PI * FREQ.THETA * t + (this.state.phases[base + 1] ?? 0));

    // ── Alpha (10 Hz, 20 μV) ─────────────────────────────────────────────
    // Pico del ritmo de Berger. 20 μV: amplitud típica en occipital con
    // ojos cerrados. Constante en este simulador (el protocolo es siempre
    // con ojos cerrados, así que no se modela el blocking alfa).
    const alpha = AMP_BASE.ALPHA *
      Math.sin(2 * Math.PI * FREQ.ALPHA * t + (this.state.phases[base + 2] ?? 0));

    // ── Beta (20 Hz, 10 μV) ──────────────────────────────────────────────
    // Beta medio. 10 μV es el nivel basal de alerta en reposo frontocentral.
    // Constante aquí; para mayor realismo en trance podría bajar a 5 μV.
    // El TBR = theta/beta; si theta sube a 80 μV → TBR = 8 → feedback potente.
    const beta = AMP_BASE.BETA *
      Math.sin(2 * Math.PI * FREQ.BETA * t + (this.state.phases[base + 3] ?? 0));

    // ── Artefacto de red eléctrica (50 Hz, 15 μV) ────────────────────────
    // Omnipresente en EEG sin jaula de Faraday. Senoidal pura (la interferencia
    // real es muy coherente). 15 μV moderado: el filtro notch debe atenuarla
    // >40 dB. Fases distintas por canal simulan pequeñas diferencias de
    // acoplamiento inductivo entre cables de electrodo.
    const lineNoise = AMP_BASE.LINE *
      Math.sin(2 * Math.PI * FREQ.LINE * t + (this.state.phases[base + 4] ?? 0));

    // ── Ruido gaussiano (σ = 5 μV) ───────────────────────────────────────
    // Actividad cortical asincrónica → espectro plano (ruido blanco).
    // Sin este componente la FFT mostraría picos perfectamente discretos,
    // lo que no ocurre en datos reales y daría falsa precisión en los tests.
    const noise = gaussianNoise(AMP_BASE.NOISE);

    return delta + theta + alpha + beta + lineNoise + noise;
  }

  /**
   * Actualiza thetaAmplitude siguiendo la rampa lineal del modo trance.
   * Usa lerp() con progreso = Δt / DURATION_MS ∈ [0, 1].
   * Cuando la rampa completa, desactiva tranceModeActive para no repetir el log.
   */
  private updateTranceRamp(): void {
    if (!this.state.tranceModeActive) return;

    const elapsed  = Date.now() - this.state.tranceModeStart;
    const progress = elapsed / TRANCE_MODE.DURATION_MS;

    this.state.thetaAmplitude = lerp(
      TRANCE_MODE.THETA_START,
      TRANCE_MODE.THETA_END,
      progress
    );

    if (progress >= 1) {
      this.state.tranceModeActive = false; // rampa completada
      console.log(
        `[EEGSimulator] Rampa trance completada. ` +
        `Theta estabilizado en ${TRANCE_MODE.THETA_END} μV.`
      );
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // FACTORY
  // ─────────────────────────────────────────────────────────────────────────────

  /**
   * Construye un EEGSimulator desde un SessionConfig.
   * Conveniente para pruebas E2E que ya tienen el config de sesión:
   *
   * @example
   *   const sim = EEGSimulator.fromSessionConfig(sessionConfig);
   *   sim.on('sample', pipeline.process.bind(pipeline));
   *   sim.start();
   */
  static fromSessionConfig(config: SessionConfig): EEGSimulator {
    return new EEGSimulator({
      sampleRate: config.sampleRate,
      channels:   config.channels,
    });
  }
}
