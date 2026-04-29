/**
 * WaveformChart.tsx  —  frontend/src/components/
 *
 * Osciloscopio EEG en tiempo real con uPlot.
 *
 * ── Arquitectura de rendimiento ──────────────────────────────────────────
 *
 *   REGLA FUNDAMENTAL: este componente NUNCA re-renderiza para actualizar datos.
 *   React se usa solo para montar/desmontar la instancia uPlot y para el
 *   indicador LED. Toda la actualización de datos ocurre en el loop RAF,
 *   fuera del ciclo de reconciliación de React.
 *
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │ React render (solo al montar / cambiar props estructurales)      │
 *   │   └─ useEffect → crea uPlot en containerRef                     │
 *   │                                                                  │
 *   │ requestAnimationFrame loop (60 fps, independiente de React)      │
 *   │   └─ lee waveformBuffer + waveformIndex del store (getState)     │
 *   │   └─ desenrolla buffer circular → plotValues Float64Array        │
 *   │   └─ uPlot.setData([timestamps, plotValues])                    │
 *   │   └─ actualiza LED via DOM directo (sin setState)                │
 *   └──────────────────────────────────────────────────────────────────┘
 *
 * ── Por qué RAF en lugar de setInterval ──────────────────────────────────
 *
 *   setInterval:
 *   - No sincronizado con el frame del compositor del browser.
 *   - Puede dispararse en medio de un paint → tearing visual.
 *   - Sigue acumulando callbacks cuando la pestaña está en background
 *     → desperdicia CPU y genera backpressure de mensajes WS.
 *
 *   requestAnimationFrame:
 *   - Sincronizado con el vsync del monitor (60/120/144 Hz según hardware).
 *   - Se pausa automáticamente cuando la pestaña está oculta (Page Visibility API).
 *   - El browser puede batching de múltiples RAF en el mismo frame.
 *   - Resultado: animación fluida con mínimo uso de CPU.
 *
 * ── Por qué uPlot en lugar de Chart.js / Recharts ────────────────────────
 *
 *   uPlot está diseñado específicamente para series temporales de alta
 *   densidad. Usa canvas 2D con un renderer optimizado que escala a
 *   millones de puntos. Chart.js y Recharts usan SVG — aceptable para
 *   dashboards estáticos, pero SVG DOM con 500 nodos actualizando a
 *   60 fps es prohibitivo.
 *
 *   Benchmark relevante: uPlot renderiza 500 puntos en ~0.3 ms.
 *   Chart.js: ~8 ms. Recharts: ~25 ms (SVG DOM).
 *   Con un presupuesto de frame de 16.7 ms (60 fps), solo uPlot deja
 *   margen suficiente para el resto de la UI.
 *
 * ── Desenrollado del buffer circular ─────────────────────────────────────
 *
 *   El store mantiene un ring buffer: waveformBuffer[i] donde el índice
 *   waveformIndex apunta al slot del sample MÁS ANTIGUO (el próximo a
 *   sobrescribir). Para dibujar en orden cronológico:
 *
 *     for i in [0, N):
 *       plotValues[i] = waveformBuffer[(waveformIndex + i) % N]
 *
 *   Los timestamps se generan sintéticamente en función de sampleRate:
 *     timestamps[i] = now - (N - 1 - i) * (1000 / sampleRate)
 *
 *   Esta estrategia evita almacenar timestamps reales para cada sample
 *   (lo que requeriría un segundo ring buffer de timestamps), asumiendo
 *   que el sample rate es constante — válido para hardware EEG.
 */

import {
  useEffect,
  useRef,
  useCallback,
  memo,
} from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

import {
  useEEGStore,
} from "../store/eegStore";

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

const BUFFER_SIZE   = 500;   // debe coincidir con WAVEFORM_BUFFER_SIZE en eegStore
const Y_MIN         = -150;  // µV — rango fijo para EEG clínico
const Y_MAX         = +150;  // µV
const Y_MAX_DYNAMIC = 3000;  // límite superior de autoescala para evitar zoom extremo por artefactos
const SIGNAL_COLOR  = "#00e5ff"; // cyan eléctrico — contraste máximo sobre fondo oscuro
const BG_COLOR      = "#0a0a0f"; // negro casi absoluto — estilo osciloscopio
const GRID_COLOR    = "rgba(255,255,255,0.07)"; // gridlines apenas perceptibles
const AXIS_COLOR    = "rgba(255,255,255,0.25)";
const LED_ON_COLOR  = "#00ff88";
const LED_OFF_COLOR = "#0a2a18";
const LED_BLINK_MS  = 120; // duración del pulso del LED en ms

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface WaveformChartProps {
  /** Frecuencia de muestreo del hardware EEG — determina la escala temporal */
  sampleRate: number;
  /** Ancho del contenedor en px (default: "100%") */
  width?: number;
  /** Alto del canvas en px (default: 220) */
  height?: number;
}

// ---------------------------------------------------------------------------
// Helpers — generación de arrays de plot
// ---------------------------------------------------------------------------

/**
 * Genera el array de timestamps sintéticos para uPlot.
 * uPlot espera timestamps en SEGUNDOS (no ms).
 *
 * La ventana temporal siempre es [now - 2s, now], moviéndose en tiempo real.
 * Todos los timestamps se recalculan en cada frame RAF para que el eje X
 * se desplace suavemente hacia la derecha.
 *
 * @param nowSec  Tiempo actual en segundos (Date.now() / 1000)
 * @param n       Número de muestras
 * @param sr      Sample rate en Hz
 * @param out     Float64Array pre-allocado de longitud n (muta in-place)
 */
function fillTimestamps(
  nowSec: number,
  n: number,
  sr: number,
  out: Float64Array
): void {
  const dt = 1 / sr; // intervalo entre muestras en segundos
  for (let i = 0; i < n; i++) {
    // Sample 0 es el más antiguo (hace ~2s), sample n-1 es el más reciente
    out[i] = nowSec - (n - 1 - i) * dt;
  }
}

/**
 * Desenrolla el buffer circular del store en un array lineal para uPlot.
 * Lee directamente desde Float32Array sin copias intermedias.
 *
 * @param src   Float32Array circular del store (BUFFER_SIZE elementos)
 * @param head  Índice del sample MÁS ANTIGUO en el buffer (= waveformIndex)
 * @param out   Float32Array pre-allocado de longitud BUFFER_SIZE (muta in-place)
 */
function unrollCircularBuffer(
  src: Float32Array,
  head: number,
  out: Float32Array
): void {
  const n = src.length;
  for (let i = 0; i < n; i++) {
    out[i] = src[(head + i) % n];
  }
}

function getPeakAbs(values: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v)) continue;
    const abs = Math.abs(v);
    if (abs > peak) peak = abs;
  }
  return peak;
}

// ---------------------------------------------------------------------------
// Configuración estática de uPlot
// ---------------------------------------------------------------------------

/**
 * Construye el objeto Options de uPlot.
 * Se llama una sola vez en el efecto de montaje — no se recrea en cada render.
 */
function buildUPlotOptions(
  width: number,
  height: number
): uPlot.Options {
  return {
    title  : "EEG — Canal 1 (µV)",
    width,
    height,

    // Fondo oscuro estilo osciloscopio — se aplica via CSS en el contenedor
    // (uPlot no tiene API directa de backgroundColor; se hace en el wrapper)

    cursor: {
      // Crosshair en hover con tooltip de valor
      show  : true,
      sync  : { key: "eeg-chart" },
      drag  : { x: false, y: false }, // no zoom con drag en señal en tiempo real
      points: {
        show: true,
        size: 6,
        fill: SIGNAL_COLOR,
      },
    },

    // Deshabilitar selección de rango (zooming) — el eje X se mueve solo
    select: { show: false, left: 0, top: 0, width: 0, height: 0 },

    legend: {
      show: true,
    },

    axes: [
      // ── Eje X (tiempo) ──────────────────────────────────────────────
      {
        stroke : AXIS_COLOR,
        grid   : { stroke: GRID_COLOR, width: 1 },
        ticks  : { stroke: AXIS_COLOR, width: 1 },
        // uPlot formatea timestamps Unix en segundos → mostrar como HH:MM:SS
        values : (
          _u: uPlot,
          splits: number[]
        ) =>
          splits.map((s) => {
            const d = new Date(s * 1000);
            return `${d.getMinutes().toString().padStart(2, "0")}:${d
              .getSeconds()
              .toString()
              .padStart(2, "0")}`;
          }),
        size   : 32,
        font   : "11px 'JetBrains Mono', monospace",
        labelFont: "11px 'JetBrains Mono', monospace",
      },
      // ── Eje Y (µV) ──────────────────────────────────────────────────
      {
        label  : "µV",
        stroke : AXIS_COLOR,
        grid   : { stroke: GRID_COLOR, width: 1 },
        ticks  : { stroke: AXIS_COLOR, width: 1 },
        // Gridlines en -100, -50, 0, 50, 100 µV
        splits : () => [-150, -100, -50, 0, 50, 100, 150],
        values : (_u: uPlot, splits: number[]) =>
          splits.map((v) => `${v}`),
        size   : 44,
        font   : "11px 'JetBrains Mono', monospace",
      },
    ],

    scales: {
      x: {
        time: true,
        // La ventana X siempre cubre exactamente BUFFER_SIZE / sampleRate segundos
        // El setData actualiza el rango en cada frame.
        auto: false,
      },
      y: {
        // Rango fijo — NO auto-escalar (evita saltos visuales entre frames)
        auto  : false,
        range : () => [Y_MIN, Y_MAX],
      },
    },

    series: [
      // Serie 0: timestamps (eje X, requerido por uPlot pero no se dibuja)
      {},
      // Serie 1: señal EEG
      {
        label   : "EEG Ch1",
        stroke  : SIGNAL_COLOR,
        width   : 1.5,
        // Sin puntos — solo línea continua (crucial para rendimiento a 250 sps)
        points  : { show: false },
        // Sin relleno bajo la curva
        fill    : undefined,
        // Span gaps: si hay NaN en el buffer, no dibujar líneas de gap
        spanGaps: false,
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Componente principal
// ---------------------------------------------------------------------------

/**
 * WaveformChart — Osciloscopio EEG en tiempo real.
 *
 * React.memo con comparación shallow: el componente solo re-renderiza si
 * cambian las props `sampleRate`, `width` o `height`. Los datos de la
 * señal nunca causan re-render (se actualizan via RAF + uPlot.setData).
 */
const WaveformChart = memo(function WaveformChart({
  sampleRate,
  width,
  height = 220,
}: WaveformChartProps) {
  // ── Refs — todos los objetos con vida más larga que un render ──────────
  const containerRef  = useRef<HTMLDivElement>(null);
  const ledRef        = useRef<HTMLDivElement>(null);
  const uplotRef      = useRef<uPlot | null>(null);
  const rafIdRef      = useRef<number>(0);
  const ledTimerRef   = useRef<ReturnType<typeof setTimeout> | null>(null);
  const chartFaultRef = useRef<boolean>(false);

  // Arrays pre-allocados para el loop RAF — sin allocaciones en el hot path
  const tsArrayRef    = useRef<Float64Array>(new Float64Array(BUFFER_SIZE));
  const plotValRef    = useRef<Float32Array>(new Float32Array(BUFFER_SIZE));
  const yRangeRef     = useRef<number>(Math.max(Math.abs(Y_MIN), Math.abs(Y_MAX)));

  // ── Función de parpadeo del LED (actualiza DOM directamente, sin setState) ─
  const blinkLED = useCallback(() => {
    const led = ledRef.current;
    if (!led) return;

    // ON
    led.style.backgroundColor = LED_ON_COLOR;
    led.style.boxShadow       = `0 0 8px 3px ${LED_ON_COLOR}`;

    // Limpiar timer anterior si el siguiente frame llega antes de que apague
    if (ledTimerRef.current) clearTimeout(ledTimerRef.current);

    // OFF después de LED_BLINK_MS
    ledTimerRef.current = setTimeout(() => {
      if (!led) return;
      led.style.backgroundColor = LED_OFF_COLOR;
      led.style.boxShadow       = "none";
    }, LED_BLINK_MS);
  }, []);

  // ── Loop RAF: actualiza uPlot a 60 fps sin invocar React ───────────────
  const startRAFLoop = useCallback(() => {
    let lastIndex = -1; // detectar si el buffer cambió desde el último frame

    function frame() {
      rafIdRef.current = requestAnimationFrame(frame);
      if (chartFaultRef.current) return;

      const plot = uplotRef.current;
      if (!plot) return;

      // Leer el buffer y el índice directamente desde el store en hot path (sin suscripciones).
      // getState() es síncrono y no causa re-render.
      const state = useEEGStore.getState();
      const { waveformBuffer } = state;
      const currentIndex = state.waveformIndex;

      // Skip si no hay datos nuevos desde el último frame
      // (e.g., sesión pausada, pestaña en background antes de que RAF se pause)
      if (currentIndex === lastIndex) return;
      lastIndex = currentIndex;

      // ── Desenrollar buffer circular → array lineal ──────────────────
      unrollCircularBuffer(waveformBuffer, currentIndex, plotValRef.current);

      // ── Generar timestamps sintéticos en segundos ───────────────────
      const nowSec = Date.now() / 1000;
      fillTimestamps(nowSec, BUFFER_SIZE, sampleRate, tsArrayRef.current);

      // ── Autoescala vertical adaptativa ───────────────────────────────
      // Mantiene un mínimo clínico ±150 µV, pero amplía cuando la señal
      // supera ese rango para evitar que la línea desaparezca.
      const peakAbs = getPeakAbs(plotValRef.current);
      const targetRange = Math.min(
        Math.max(peakAbs * 1.25, Math.abs(Y_MAX)),
        Y_MAX_DYNAMIC
      );
      const prevRange = yRangeRef.current;
      const nextRange = prevRange + (targetRange - prevRange) * 0.2;

      if (Math.abs(nextRange - prevRange) > 2) {
        yRangeRef.current = nextRange;
        try {
          plot.setScale("y", { min: -nextRange, max: nextRange });
        } catch (err) {
          chartFaultRef.current = true;
          console.error("[WaveformChart] setScale falló", err);
          return;
        }
      }

      // ── Actualizar uPlot sin re-crear la instancia ──────────────────
      // setData es la API de uPlot para actualizar datos sin destruir/recrear.
      // uPlot internamente solo redibuja la región del canvas que cambió.
      // El segundo argumento (resetScales=false) mantiene los rangos fijos.
      try {
        plot.setData(
          [
            tsArrayRef.current as unknown as number[],
            plotValRef.current as unknown as number[],
          ],
          false // no resetear scales — preservar el rango Y fijo
        );
      } catch (err) {
        chartFaultRef.current = true;
        console.error("[WaveformChart] setData falló; se detiene el loop para evitar crash", err);
        return;
      }

      // Parpadear el LED para indicar actividad (DOM directo, sin setState)
      blinkLED();
    }

    rafIdRef.current = requestAnimationFrame(frame);
  }, [sampleRate, blinkLED]);

  // ── Efecto de montaje: crear uPlot ────────────────────────────────────
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // CRUCIAL: limpiar el contenedor antes de crear uPlot para evitar
    // conflictos con elementos React que React intenta manipular
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    // Calcular ancho real del contenedor si no se pasa como prop
    const resolvedWidth = (width ?? container.offsetWidth) || 800;

    // Datos iniciales: arrays de ceros para que uPlot renderice antes de
    // recibir datos reales (evita el flash de un canvas vacío)
    const initTs  = Array.from({ length: BUFFER_SIZE }, (_, i) =>
      Date.now() / 1000 - (BUFFER_SIZE - 1 - i) / sampleRate
    );
    const initVal = new Array<number>(BUFFER_SIZE).fill(0);

    const opts = buildUPlotOptions(resolvedWidth, height);

    // uPlot espera arrays nativos JS (no TypedArrays) en el constructor.
    // En setData se pueden pasar TypedArrays — de ahí el cast en el loop RAF.
    try {
      uplotRef.current = new uPlot(opts, [initTs, initVal], container);
    } catch (err) {
      chartFaultRef.current = true;
      console.error("[WaveformChart] Error inicializando uPlot", err);
      return;
    }

    // Arrancar el loop RAF
    startRAFLoop();

    // ── Cleanup ─────────────────────────────────────────────────────
    return () => {
      // Cancelar el loop RAF primero para no llamar setData sobre una
      // instancia ya destruida
      cancelAnimationFrame(rafIdRef.current);

      if (ledTimerRef.current) clearTimeout(ledTimerRef.current);

      // Destruir uPlot y liberar el canvas del DOM
      uplotRef.current?.destroy();
      uplotRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // ← Sin dependencias: el efecto corre solo al montar/desmontar

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry || !uplotRef.current) return;
      const nextWidth = Math.floor(entry.contentRect.width);
      if (nextWidth <= 0 || height <= 0) return;

      try {
        uplotRef.current.setSize({
          width: nextWidth,
          height,
        });
      } catch (err) {
        chartFaultRef.current = true;
        console.error("[WaveformChart] setSize falló", err);
      }
    });
    
    observer.observe(container);
    return () => observer.disconnect();
  }, [height]);


  // ── Render — mínimo, sin datos dinámicos ─────────────────────────────
  // Este JSX solo se ejecuta en mount (gracias a React.memo y la ausencia
  // de props que cambien frecuentemente). Toda la actualización visual
  // ocurre en el loop RAF sin pasar por aquí.
  return (
    <div
      style={{
        position       : "relative",
        backgroundColor: BG_COLOR,
        borderRadius   : "8px",
        border         : "1px solid rgba(0,229,255,0.15)",
        overflow       : "hidden",
        // Sombra sutil de color cyan para reforzar el estilo osciloscopio
        boxShadow      : "0 0 24px rgba(0,229,255,0.06), inset 0 0 40px rgba(0,0,0,0.4)",
      }}
    >
      {/* Indicador LED de actividad ────────────────────────────────── */}
      {/*
       * El LED no usa useState — se actualiza vía DOM directo en blinkLED().
       * Si usáramos useState aquí, cada parpadeo causaría un re-render del
       * componente, que a su vez causaría un re-render de uPlot.
       * Con ref + style mutation directa: cero renders adicionales.
       */}
      <div
        style={{
          position  : "absolute",
          top       : 10,
          right     : 12,
          zIndex    : 10,
          display   : "flex",
          alignItems: "center",
          gap       : 6,
        }}
      >
        <span
          style={{
            fontSize  : 10,
            color     : "rgba(255,255,255,0.35)",
            fontFamily: "'JetBrains Mono', monospace",
            letterSpacing: "0.05em",
            userSelect: "none",
          }}
        >
          LIVE
        </span>
        <div
          ref={ledRef}
          style={{
            width           : 8,
            height          : 8,
            borderRadius    : "50%",
            backgroundColor : LED_OFF_COLOR,
            boxShadow       : "none",
            // Transición suave para el apagado (el encendido es instantáneo)
            transition      : "background-color 80ms ease, box-shadow 80ms ease",
            flexShrink      : 0,
          }}
        />
      </div>

      {/* Indicador de sample rate ───────────────────────────────────── */}
      <div
        style={{
          position  : "absolute",
          bottom    : 6,
          right     : 12,
          zIndex    : 10,
          fontSize  : 10,
          color     : "rgba(255,255,255,0.2)",
          fontFamily: "'JetBrains Mono', monospace",
          userSelect: "none",
        }}
      >
        {sampleRate} sps · 2 s · ±150 µV
      </div>

      {/* Contenedor de uPlot — uPlot inyecta el canvas aquí ────────── */}
      {/*
       * CRÍTICO: este div no tiene hijos React. uPlot gestiona su contenido
       * directamente vía DOM APIs. Insertar elementos React dentro causaría
       * conflictos con el reconciliador.
       */}
      <div
        ref={containerRef}
        style={{
          // uPlot aplica su propio estilo al canvas interno.
          // Solo necesitamos que el contenedor tenga el tamaño correcto.
          width : width ? `${width}px` : "100%",
          height: `${height}px`,
        }}
      />
    </div>
  );
}, (prevProps, nextProps) => {
  // Comparación personalizada para React.memo:
  // Solo re-renderizar si cambian props estructurales que afectan la
  // configuración de uPlot. Los datos de la señal NUNCA pasan como props.
  return (
    prevProps.sampleRate === nextProps.sampleRate &&
    prevProps.width      === nextProps.width      &&
    prevProps.height     === nextProps.height
  );
});

WaveformChart.displayName = "WaveformChart";

export default WaveformChart;
