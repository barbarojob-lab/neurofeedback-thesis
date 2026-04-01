/**
 * @file src/types.ts
 * @description Tipos centrales del sistema de neurofeedback EEG en tiempo real.
 *
 * ── Pipeline de datos ────────────────────────────────────────────────────────
 *
 *   Dispositivo EEG
 *       │
 *       ▼
 *   EEGSample  ──── muestra cruda por canal
 *       │
 *       ▼
 *   (Filtrado: notch 50/60 Hz, bandpass 0.5–45 Hz)
 *       │
 *       ▼
 *   BandPowers  ──── FFT por ventana (~2 s, 50 % overlap)
 *       │
 *       ├──► ThetaBetaResult  ──── biomarcador primario θ/β
 *       │
 *       └──► ZScoreResult  ──────── normalización individual
 *                │
 *                ▼
 *           FeedbackCommand  ──── decisión de feedback
 *                │
 *                ▼
 *           FeedbackPayload  ──── envelope completo al cliente
 *                │
 *                ▼
 *           WSMessage  ─────────── sobre WebSocket
 *
 * ── Par dual neurofisiológico-fenomenológico ─────────────────────────────────
 *
 *   ThetaBetaResult + ZScoreResult  ←──────────────►  SubjectiveMeasure
 *        (polo EEG / objetivo)                         (polo fenomenológico)
 *        Medida continua ~4 Hz                         NRS-T + AOES puntual
 *
 *   Este par permite validar que el biomarcador EEG tiene correlato
 *   experiencial, y detectar disociaciones (EEG elevado sin reporte,
 *   o reporte elevado sin cambio EEG).
 */

// ─────────────────────────────────────────────────────────────────────────────
// 1. ADQUISICIÓN DE SEÑAL
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Muestra EEG cruda tal como llega del dispositivo (OpenBCI, Emotiv, Muse…).
 * Es la **unidad atómica** que recorre todo el pipeline; cada callback
 * del driver produce un EEGSample por tick de reloj del hardware.
 */
export interface EEGSample {
  /**
   * Instante de captura en ms desde epoch Unix (Date.now() en el driver).
   * Referencia temporal para sincronizar con eventos externos (marcadores
   * de estímulo, inicio de bloque) y con SubjectiveMeasure.timestamp.
   */
  timestamp: number;

  /**
   * Voltajes de cada canal en microvoltios (μV), almacenados como Float32Array.
   * Float32 reduce la memoria a la mitad respecto a Float64 sin perder
   * resolución relevante para EEG (±500 μV, ruido ~0.1 μV).
   * Orden de canales: según montaje 10-20 configurado en SessionConfig
   * (p. ej. índice 0 → Fz, 1 → Cz, 2 → Pz para protocolo de línea media).
   */
  channelData: Float32Array;

  /**
   * Frecuencia de muestreo del hardware en Hz.
   * Determina la frecuencia de Nyquist (sampleRate / 2) y el tamaño
   * de ventana FFT necesario para resolución espectral de 0.5 Hz en theta.
   */
  sampleRate: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. ANÁLISIS ESPECTRAL
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Potencias espectrales por banda clásica, calculadas mediante FFT
 * sobre una ventana deslizante (típicamente 2 s, solapamiento 50 %).
 *
 * Unidades: μV² (potencia absoluta) o sin unidad si se normaliza
 * como fracción de la potencia total (potencia relativa).
 */
export interface BandPowers {
  /**
   * Delta (0.5–4 Hz): ondas de máxima amplitud.
   * En sujetos despiertos indica artefactos de movimiento o somnolencia extrema.
   * Monitorizado para detección de artefactos y control de calidad de señal.
   */
  delta: number;

  /**
   * Theta (4–8 Hz): **biomarcador primario del sistema**.
   * Se potencia durante meditación profunda, estados hipnóticos y trance ligero.
   * El protocolo busca maximizar theta en el grupo experimental mediante reward.
   */
  theta: number;

  /**
   * Alpha (8–12 Hz): marcador de relajación ocular e inhibición cortical.
   * Co-varía con theta en estados contemplativos; su aumento puede indicar
   * transición hacia trance o simplemente relajación basal con ojos cerrados.
   */
  alpha: number;

  /**
   * Beta (13–30 Hz): procesamiento cognitivo activo y alerta cortical.
   * **Denominador del ratio θ/β**: su disminución relativa es parte del objetivo.
   * Un beta elevado durante la sesión puede indicar ansiedad o fuga cognitiva.
   */
  beta: number;

  /**
   * Gamma (>30 Hz): procesamiento de alto nivel e integración sensorial.
   * Monitorizado para detección de artefactos musculares (EMG contamina >35 Hz)
   * y como exploración secundaria; no es objetivo primario del protocolo.
   */
  gamma: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. BIOMARCADOR PRIMARIO — RATIO THETA / BETA
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Resultado del cálculo del ratio Theta/Beta (TBR), índice neurofisiológico
 * central del protocolo.
 *
 * Fundamento: TBR > 1 indica predominancia de actividad lenta (theta)
 * sobre actividad rápida (beta) — asociado a estados de trance, hipnosis
 * y meditación profunda (Isotani et al., 2001; Gruzelier, 2014).
 *
 * Se calcula por epoch y se acumula en la ventana de ZScore para
 * normalización individual.
 */
export interface ThetaBetaResult {
  /**
   * Ratio θ/β = thetaPower / betaPower.
   * El motor de feedback compara este valor con el umbral dinámico
   * derivado del ZScore para generar FeedbackCommand.
   * Rango típico en sujetos en reposo: 0.5–3.0.
   */
  ratio: number;

  /**
   * Potencia theta del epoch actual (μV²).
   * Numerador del ratio; se almacena por separado para análisis post-hoc
   * de cada componente individual.
   */
  thetaPower: number;

  /**
   * Potencia beta del epoch actual (μV²).
   * Denominador del ratio; útil para detectar sesiones con beta anormalmente
   * bajo (somnolencia) que inflarían artificialmente el TBR.
   */
  betaPower: number;

  /**
   * Timestamp del epoch analizado (ms desde epoch Unix).
   * Permite alinear TBR con SubjectiveMeasure.timestamp en análisis de
   * correlación temporal y con marcadores de eventos de la sesión.
   */
  timestamp: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. NORMALIZACIÓN ESTADÍSTICA — Z-SCORE
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Z-score del TBR (o de la potencia theta) respecto a una ventana deslizante
 * de baseline individual.
 *
 * Propósito: los valores absolutos de EEG varían enormemente entre sujetos
 * (anatomía craneal, impedancia de electrodos, medicación…). Normalizar
 * al baseline de cada sujeto garantiza que el umbral de feedback sea
 * **relativo a su propio estado basal**, no a una norma poblacional.
 *
 * zScore = (valor_actual − media_ventana) / std_ventana
 */
export interface ZScoreResult {
  /**
   * Z-score del epoch actual respecto al baseline de la ventana.
   * Valor > +1.5 → activación theta significativamente por encima del baseline
   *   → FeedbackCommand: 'increase_theta' (reward).
   * Valor < −1.0 → caída por debajo del baseline
   *   → FeedbackCommand: 'decrease_theta' (alerta suave).
   * Los umbrales exactos se parametrizan en SessionConfig o en el motor de reglas.
   */
  zScore: number;

  /**
   * Media aritmética del TBR en la ventana de baseline (últimas `windowSize` muestras).
   * Se actualiza en cada epoch para que el baseline se adapte lentamente
   * a la deriva fisiológica de la sesión.
   */
  mean: number;

  /**
   * Desviación estándar del TBR en la ventana de baseline.
   * Un std muy bajo indica señal estable (buena sesión) o artefacto
   * de saturación (señal plana — requiere comprobación de impedancias).
   */
  std: number;

  /**
   * Número de epochs usados para calcular media y std.
   * Ventana larga → estimación más robusta pero adaptación más lenta
   *   a cambios de estado genuinos.
   * Valor típico: 60 epochs = 2 min de sesión a epoch cada 2 s.
   */
  windowSize: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. LÓGICA DE FEEDBACK
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Comando de feedback generado por el motor de decisión tras evaluar
 * el ZScore del epoch actual.
 *
 * Se traduce en el cliente a un estímulo concreto:
 *   - Auditivo: volumen / pitch de un tono continuo.
 *   - Visual: brillo / tamaño de una figura en pantalla.
 *   - Háptico: intensidad de vibración (wearable).
 */
export interface FeedbackCommand {
  /**
   * Acción de feedback determinada por el motor de reglas:
   *
   * 'increase_theta' — zScore > umbral_alto
   *   El sujeto está produciendo más theta de lo habitual → recompensar.
   *   El estímulo (sonido, luz) se hace más agradable / intenso.
   *
   * 'decrease_theta' — zScore < umbral_bajo
   *   El sujeto ha caído por debajo del baseline → señal de re-enfoque.
   *   El estímulo se atenúa o cambia de carácter.
   *
   * 'neutral' — zScore dentro del rango aceptable
   *   No se interviene; el estímulo mantiene su estado actual.
   */
  action: 'increase_theta' | 'decrease_theta' | 'neutral';

  /**
   * Intensidad de la intervención en rango continuo [0, 1].
   * Proporcional al zScore: permite feedback gradual en lugar de binario.
   * 0 = cambio mínimo perceptible (calibrado por sujeto en baseline).
   * 1 = máximo configurado en la sesión (para evitar saturación sensorial).
   *
   * Ejemplo de mapeo: intensity = clamp((|zScore| − threshold) / range, 0, 1)
   */
  intensity: number; // [0, 1]
}

/**
 * Payload completo enviado al cliente vía WebSocket en cada ciclo de feedback
 * (~4 Hz con ventana de 2 s y solapamiento del 50 %).
 *
 * Agrega todos los productos del pipeline para que el cliente pueda:
 *   (a) Ejecutar el feedback en tiempo real (comando + intensidad).
 *   (b) Renderizar el dashboard clínico (señal filtrada, potencias, ratio).
 *   (c) Registrar el estado completo para análisis post-hoc.
 */
export interface FeedbackPayload {
  /**
   * Discriminador de tipo para el enrutamiento en el handler WebSocket del cliente.
   * Otros tipos posibles en WSMessage: 'session_start', 'session_end', 'error', 'ping'.
   */
  type: 'eeg_data';

  /**
   * Timestamp del ciclo de procesamiento (ms desde epoch Unix).
   * Referencia temporal del servidor; puede diferir levemente del
   * EEGSample.timestamp por latencia de procesamiento (~5–20 ms).
   */
  timestamp: number;

  /**
   * Muestra EEG tras el pipeline de filtrado digital:
   *   1. Filtro notch 50/60 Hz (elimina interferencia de red eléctrica).
   *   2. Filtro paso-banda 0.5–45 Hz (elimina deriva DC y artefactos de alta frecuencia).
   *   3. (Opcional) ICA / ASR para eliminación de artefactos oculares y musculares.
   * Se envía para visualización en trazado EEG del dashboard clínico.
   */
  filteredSample: EEGSample;

  /**
   * Potencias espectrales del epoch actual para monitorización multiband.
   * Permite al clínico observar la evolución de todas las bandas en tiempo real,
   * no solo el ratio objetivo.
   */
  bandPowers: BandPowers;

  /**
   * Resultado del biomarcador primario θ/β para el epoch actual.
   * El cliente lo registra en la base de datos de sesión y lo muestra
   * en la gráfica de TBR temporal.
   */
  thetaBeta: ThetaBetaResult;

  /**
   * Z-score normalizado al baseline individual del sujeto.
   * Base matemática sobre la que se genera el FeedbackCommand.
   * Se almacena para reconstruir la lógica de decisión en el análisis post-hoc.
   */
  zScore: ZScoreResult;

  /**
   * Comando de feedback listo para ejecutar.
   * El cliente lo consume directamente sin lógica adicional:
   *   audioEngine.setVolume(command.intensity);  // si action !== 'neutral'
   */
  command: FeedbackCommand;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. CONFIGURACIÓN DE SESIÓN
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Configuración inmutable de una sesión experimental.
 * Se crea al iniciar la sesión, se persiste en la base de datos y
 * acompaña todos los registros (FeedbackPayload, SubjectiveMeasure)
 * como metadato de contexto experimental.
 */
export interface SessionConfig {
  /**
   * Identificador anónimo del participante (p. ej. "P01", "EXP_042").
   * ⚠ NUNCA debe contener datos personales identificables.
   * Cumplimiento GDPR / ética de investigación: el mapeo ID→persona
   * se mantiene en documento separado bajo acceso restringido.
   */
  patientId: string;

  /**
   * Asignación de grupo experimental:
   *
   * 'experimental': recibe feedback neurofisiológico real (TBR en tiempo real).
   * 'control':      recibe feedback sham — señal EEG aleatorizada o retrasada
   *                 ~20 minutos para desacoplar el feedback del estado actual.
   *
   * La asignación determina qué lógica ejecuta el motor de feedback
   * (FeedbackEngine.process() vs ShamEngine.process()).
   */
  group: 'experimental' | 'control';

  /**
   * Frecuencia de muestreo del dispositivo EEG en Hz.
   * 250 Hz: resolución estándar; suficiente para theta (4–8 Hz) y beta (13–30 Hz).
   * 500 Hz: necesario si se analiza gamma (>30 Hz) o se aplica ICA online.
   * El tipo literal evita configuraciones inválidas en tiempo de compilación.
   */
  sampleRate: 250 | 500;

  /**
   * Número de canales EEG activos en el montaje.
   * Determina la longitud de EEGSample.channelData y el coste computacional
   * de la FFT y del ICA (si se usa).
   * Valores típicos: 8 (OpenBCI Cyton), 16 (Cyton+Daisy), 4 (Muse 2).
   */
  channels: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. PROTOCOLO WEBSOCKET
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Envelope genérico para todos los mensajes del protocolo WebSocket.
 *
 * Permite multiplexar distintos tipos de mensajes en la misma conexión
 * sin necesidad de múltiples sockets:
 *   { type: 'eeg_data',         payload: FeedbackPayload      }
 *   { type: 'subjective_measure', payload: SubjectiveMeasure  }
 *   { type: 'session_start',    payload: SessionConfig        }
 *   { type: 'session_end',      payload: { sessionId: number } }
 *   { type: 'error',            payload: { message: string }  }
 *   { type: 'ping',             payload: null                 }
 */
export interface WSMessage {
  /**
   * Discriminador de tipo del mensaje.
   * El handler WebSocket del cliente hace switch/case sobre este campo
   * para enrutar al procesador correcto.
   * Se recomienda usar un enum o union literal para estrechar el tipo
   * en implementaciones concretas.
   */
  type: string;

  /**
   * Cuerpo del mensaje con tipo desconocido en esta interfaz base.
   * Se estrecha mediante type guards o validación Zod en cada handler:
   *   if (msg.type === 'eeg_data') {
   *     const payload = FeedbackPayloadSchema.parse(msg.payload);
   *   }
   */
  payload: unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. MEDIDAS SUBJETIVAS — POLO FENOMENOLÓGICO DEL PAR DUAL
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Medida subjetiva recogida en puntos temporales predefinidos de la sesión
 * (p. ej. al final de cada bloque de 5 min, o cada vez que el sujeto
 * activa un botón de autoevaluación).
 *
 * ── Par dual neurofisiológico-fenomenológico ──────────────────────────────
 *
 *   POLO EEG (continuo ~4 Hz)           POLO FENOMENOLÓGICO (puntual)
 *   ─────────────────────────           ─────────────────────────────
 *   ThetaBetaResult.ratio          ←──► SubjectiveMeasure.nrsT
 *   ZScoreResult.zScore            ←──► SubjectiveMeasure.aoesTotalScore
 *
 * Este par permite:
 *   1. Validar que el biomarcador EEG tiene correlato experiencial real.
 *   2. Detectar disociaciones (EEG elevado sin reporte, o reporte sin EEG).
 *   3. Construir modelos de predicción cruzada (EEG → experiencia) para
 *      investigación futura sobre marcadores de estados alterados.
 *
 * Escalas utilizadas:
 *   NRS-T  — Numeric Rating Scale for Trance depth (Spiegel & Spiegel, 2004).
 *   AOES   — Absorption in Ongoing Experience Scale (Tart, revisada).
 */
export interface SubjectiveMeasure {
  /**
   * ID de la sesión a la que pertenece esta medida.
   * Foreign key hacia el registro de SessionConfig en la base de datos.
   * Permite agregar todas las medidas de una sesión en el análisis.
   */
  sessionId: number;

  /**
   * Momento de la autoevaluación (ms desde epoch Unix).
   * Se alinea con ThetaBetaResult.timestamp para correlación temporal:
   *   Δt = SubjectiveMeasure.timestamp − ThetaBetaResult.timestamp
   * Un Δt pequeño (<30 s) permite un análisis de correlación más fiable.
   */
  timestamp: number;

  /**
   * NRS-T: Numeric Rating Scale for Trance depth.
   * Escala unidimensional de 0 a 10 (respuesta verbal o táctil):
   *   0  = completamente despierto, sin alteración del estado ordinario.
   *   5  = trance moderado: atención absorta, cuerpo relajado, tiempo alterado.
   *   10 = trance profundo: máxima alteración, respuesta reducida al entorno.
   *
   * Ventaja clínica: mínimamente invasiva — el sujeto responde con un número
   * sin interrumpir significativamente el estado de trance.
   * Correlaciona con TBR en estudios de hipnosis (Isotani et al., 2001).
   */
  nrsT: number; // [0, 10]

  /**
   * AOES: Absorption in Ongoing Experience Scale — 5 ítems, cada uno [0, 4].
   * Mide la calidad fenomenológica de la absorción en 5 dimensiones:
   *
   *   [0] Absorción atencional: "Mi atención estaba completamente focalizada."
   *   [1] Fusión sujeto-objeto: "La distinción entre yo y la experiencia se disolvió."
   *   [2] Control voluntario:   "Sentí que la experiencia sucedía sola, sin esfuerzo."
   *   [3] Alteración temporal:  "Perdí la noción del tiempo transcurrido."
   *   [4] Autoconciencia:       "Disminuyó mi monitorización habitual de mí mismo."
   *
   * El orden de los ítems debe respetarse para la puntuación y el análisis factorial.
   * Rango por ítem: 0 = nada, 1 = un poco, 2 = moderadamente, 3 = bastante, 4 = completamente.
   */
  aoesItems: number[]; // longitud 5, cada elemento [0, 4]

  /**
   * Puntuación total AOES = suma de los 5 ítems. Rango: [0, 20].
   * Pre-calculada para evitar recomputación en análisis estadístico:
   *   aoesTotalScore = aoesItems.reduce((a, b) => a + b, 0)
   *
   * Se correlaciona con ZScoreResult.zScore para validar el polo fenomenológico:
   *   r(aoesTotalScore, zScore) > 0.4 → validez convergente del biomarcador.
   */
  aoesTotalScore: number; // [0, 20]
}