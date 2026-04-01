/**
 * ThetaBetaGauge.tsx  —  frontend/src/components/
 *
 * Gauge SVG semicircular que visualiza el ratio Theta/Beta en tiempo real.
 *
 * ── Geometría del arco ────────────────────────────────────────────────────
 *
 *   Semicírculo: ángulo de -180° (izquierda) a 0° (derecha), medido desde
 *   el eje positivo X. El arco se dibuja con SVG path usando comandos A (arc).
 *
 *   Centro del SVG: (cx, cy) = (110, 110)
 *   Radio del arco de fondo: R = 80px
 *   Grosor del arco: strokeWidth = 18px
 *
 *   Conversión valor → ángulo:
 *     ratio ∈ [0, 3]  →  ángulo ∈ [-180°, 0°]   (de izquierda a derecha)
 *     θ(v) = -180 + (v / MAX_RATIO) × 180
 *
 * ── Gradiente de color del arco ──────────────────────────────────────────
 *
 *   0.0  → azul   #4444ff  (arousal muy alto / déficit de relajación)
 *   1.5  → verde  #00e676  (relajación normal / baseline)
 *   2.0  → lima   #aeea00  (trance ligero — zona óptima)
 *   3.0  → rojo   #ff1744  (somnolencia / trance muy profundo)
 *
 *   El color del arco activo se interpola entre estos 4 stops en función
 *   del ratio actual, usando interpolación lineal por segmento.
 *
 * ── Zona óptima [1.2, 2.0] ────────────────────────────────────────────────
 *
 *   Un arco SVG secundario (muy tenue) resalta el rango de trance óptimo.
 *   Rango clínico: TBR 1.2–2.0 = relajación profunda sin somnolencia.
 *   Por encima de 2.0 el paciente puede estar demasiado próximo al sueño.
 */

import React, { memo, useMemo } from "react";
import { useEEGStore, selectThetaBetaRatio } from "../store/eegStore";

// ---------------------------------------------------------------------------
// Constantes de geometría
// ---------------------------------------------------------------------------

const CX           = 110;   // centro X del SVG
const CY           = 108;   // centro Y (ligeramente hacia abajo para dar espacio al texto)
const R            = 80;    // radio del arco
const STROKE_W     = 18;    // grosor del trazo del arco
const MAX_RATIO    = 3.0;   // valor máximo del gauge
const OPTIMAL_LOW  = 1.2;   // inicio de zona óptima [TBR]
const OPTIMAL_HIGH = 2.0;   // fin de zona óptima [TBR]

// ---------------------------------------------------------------------------
// Helpers de geometría SVG
// ---------------------------------------------------------------------------

/** Convierte ángulo en grados → coordenadas (x, y) sobre el arco */
function polarToXY(angleDeg: number, radius: number): { x: number; y: number } {
  const rad = (angleDeg * Math.PI) / 180;
  return {
    x: CX + radius * Math.cos(rad),
    y: CY + radius * Math.sin(rad),
  };
}

/**
 * Construye el path SVG de un arco de `startDeg` a `endDeg`.
 * Los ángulos siguen la convención estándar SVG:
 *   0° = derecha, 90° = abajo, -90° (o 270°) = arriba, ±180° = izquierda
 *
 * Para el semicírculo del gauge:
 *   startDeg = 180° (extremo izquierdo)
 *   endDeg   = 0°   (extremo derecho)
 */
function arcPath(startDeg: number, endDeg: number, radius: number): string {
  const start   = polarToXY(startDeg, radius);
  const end     = polarToXY(endDeg,   radius);
  const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  // sweep-flag=1: dirección de las agujas del reloj
  return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

/**
 * Convierte un ratio [0, MAX_RATIO] a ángulo en grados para el gauge.
 * El semicírculo va de 180° (izquierda = ratio 0) a 0° (derecha = ratio 3).
 */
function ratioToAngle(ratio: number): number {
  const clamped = Math.max(0, Math.min(ratio, MAX_RATIO));
  return 180 - (clamped / MAX_RATIO) * 180;
}

// ---------------------------------------------------------------------------
// Interpolación de color
// ---------------------------------------------------------------------------

interface ColorStop {
  at: number;     // valor de ratio en este stop
  r: number; g: number; b: number;
}

const COLOR_STOPS: ColorStop[] = [
  { at: 0.0, r: 68,  g: 68,  b: 255 }, // azul
  { at: 1.5, r: 0,   g: 230, b: 118 }, // verde
  { at: 2.0, r: 174, g: 234, b: 0   }, // lima/amarillo
  { at: 3.0, r: 255, g: 23,  b: 68  }, // rojo
];

function lerpColor(ratio: number): string {
  const clamped = Math.max(0, Math.min(ratio, MAX_RATIO));

  // Encontrar el segmento de color
  let lo = COLOR_STOPS[0];
  let hi = COLOR_STOPS[COLOR_STOPS.length - 1];

  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    if (clamped >= COLOR_STOPS[i].at && clamped <= COLOR_STOPS[i + 1].at) {
      lo = COLOR_STOPS[i];
      hi = COLOR_STOPS[i + 1];
      break;
    }
  }

  const t = lo.at === hi.at ? 0 : (clamped - lo.at) / (hi.at - lo.at);
  const r = Math.round(lo.r + t * (hi.r - lo.r));
  const g = Math.round(lo.g + t * (hi.g - lo.g));
  const b = Math.round(lo.b + t * (hi.b - lo.b));

  return `rgb(${r},${g},${b})`;
}

// ---------------------------------------------------------------------------
// Longitud total del arco (para stroke-dasharray)
// ---------------------------------------------------------------------------

// Longitud del semicírculo: π × R
const ARC_LENGTH = Math.PI * R; // ≈ 251.3 px

/**
 * Calcula el stroke-dashoffset para animar el arco activo.
 * uPlot technique: usamos stroke-dasharray = [ARC_LENGTH, ARC_LENGTH]
 * y variamos stroke-dashoffset para revelar la fracción del arco.
 *
 * dashoffset = ARC_LENGTH × (1 - fracción)
 * fracción   = ratio / MAX_RATIO
 */
function ratioToDashOffset(ratio: number): number {
  const clamped = Math.max(0, Math.min(ratio, MAX_RATIO));
  const fraction = clamped / MAX_RATIO;
  return ARC_LENGTH * (1 - fraction);
}

// ---------------------------------------------------------------------------
// Componente
// ---------------------------------------------------------------------------

const ThetaBetaGauge = memo(function ThetaBetaGauge() {
  const ratio = useEEGStore(selectThetaBetaRatio);

  // Valores derivados memoizados — evitar recalcular en cada render
  const arcColor    = useMemo(() => lerpColor(ratio), [ratio]);
  const dashOffset  = useMemo(() => ratioToDashOffset(ratio), [ratio]);

  // Ángulos de la zona óptima (arco resaltado tenue)
  const optAngleHi  = 180 - (OPTIMAL_LOW  / MAX_RATIO) * 180; // 1.2 → 108°
  const optAngleLo  = 180 - (OPTIMAL_HIGH / MAX_RATIO) * 180; // 2.0 →  60°

  // Clasificación textual del estado
  const stateLabel = useMemo(() => {
    if (ratio < 0.8)  return { text: "MUY ACTIVO",   color: "#4444ff" };
    if (ratio < 1.2)  return { text: "ALERTA",        color: "#64b5f6" };
    if (ratio < 2.0)  return { text: "TRANCE ÓPTIMO", color: "#00e676" };
    if (ratio < 2.6)  return { text: "PROFUNDO",      color: "#aeea00" };
    return               { text: "SOMNOLENCIA",    color: "#ff7043" };
  }, [ratio]);

  return (
    <div style={{
      display       : "flex",
      flexDirection : "column",
      alignItems    : "center",
      userSelect    : "none",
    }}>
      <svg
        width={220}
        height={140}
        viewBox="0 0 220 140"
        aria-label={`Ratio Theta/Beta: ${ratio.toFixed(2)}`}
      >
        <defs>
          {/* Filtro de brillo para el arco activo */}
          <filter id="glow-tbr">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* ── Arco de fondo (gris oscuro, semicírculo completo) ────────── */}
        <path
          d={arcPath(180, 0, R)}
          fill="none"
          stroke="rgba(255,255,255,0.07)"
          strokeWidth={STROKE_W}
          strokeLinecap="round"
        />

        {/* ── Zona óptima [1.2, 2.0] resaltada en verde muy tenue ──────── */}
        <path
          d={arcPath(optAngleHi, optAngleLo, R)}
          fill="none"
          stroke="rgba(0,230,118,0.18)"
          strokeWidth={STROKE_W + 4}
          strokeLinecap="round"
        />
        {/* Etiqueta "ÓPTIMO" en la zona */}
        <text
          x={CX}
          y={CY - R - 6}
          textAnchor="middle"
          fill="rgba(0,230,118,0.5)"
          fontSize={8}
          fontFamily="'JetBrains Mono', monospace"
          letterSpacing="0.08em"
        >
          ÓPTIMO
        </text>

        {/* ── Arco activo (dash-offset animation) ──────────────────────── */}
        {/*
         * stroke-dasharray = [ARC_LENGTH, ARC_LENGTH]:
         *   el primer segmento es el arco visible, el segundo el hueco.
         * stroke-dashoffset controla cuánto del primer segmento se oculta.
         * Con CSS transition sobre dashoffset → animación suave.
         *
         * El arco se dibuja de izquierda (180°) a derecha (0°).
         * Para que el arco crezca de izquierda a derecha, usamos
         * pathLength + transform en lugar de stroke-dasharray directo.
         */}
        <path
          d={arcPath(180, 0, R)}
          fill="none"
          stroke={arcColor}
          strokeWidth={STROKE_W}
          strokeLinecap="round"
          strokeDasharray={`${ARC_LENGTH} ${ARC_LENGTH}`}
          strokeDashoffset={dashOffset}
          filter="url(#glow-tbr)"
          style={{
            transition: "stroke-dashoffset 0.4s ease, stroke 0.4s ease",
          }}
        />

        {/* ── Marcas de escala en 0, 1, 1.5, 2, 3 ─────────────────────── */}
        {[0, 1, 1.5, 2, 3].map((v) => {
          const ang = ratioToAngle(v);
          const outer = polarToXY(ang, R + STROKE_W / 2 + 6);
          const inner = polarToXY(ang, R - STROKE_W / 2 - 2);
          const label = polarToXY(ang, R + STROKE_W / 2 + 16);
          return (
            <g key={v}>
              <line
                x1={inner.x} y1={inner.y}
                x2={outer.x} y2={outer.y}
                stroke="rgba(255,255,255,0.3)"
                strokeWidth={1}
              />
              <text
                x={label.x}
                y={label.y + 3}
                textAnchor="middle"
                fill="rgba(255,255,255,0.35)"
                fontSize={8}
                fontFamily="'JetBrains Mono', monospace"
              >
                {v}
              </text>
            </g>
          );
        })}

        {/* ── Valor numérico central ────────────────────────────────────── */}
        <text
          x={CX}
          y={CY + 8}
          textAnchor="middle"
          fill={arcColor}
          fontSize={30}
          fontWeight={700}
          fontFamily="'JetBrains Mono', monospace"
          style={{ transition: "fill 0.4s ease" }}
        >
          {ratio.toFixed(2)}
        </text>

        {/* ── Subtítulo Θ/β ─────────────────────────────────────────────── */}
        <text
          x={CX}
          y={CY + 26}
          textAnchor="middle"
          fill="rgba(255,255,255,0.4)"
          fontSize={11}
          fontFamily="'JetBrains Mono', monospace"
          letterSpacing="0.1em"
        >
          Ratio Θ/β
        </text>
      </svg>

      {/* ── Estado textual debajo del SVG ─────────────────────────────── */}
      <div style={{
        marginTop  : -4,
        fontSize   : 10,
        fontFamily : "'JetBrains Mono', monospace",
        letterSpacing: "0.12em",
        color      : stateLabel.color,
        transition : "color 0.4s ease",
        fontWeight : 600,
      }}>
        {stateLabel.text}
      </div>
    </div>
  );
});

ThetaBetaGauge.displayName = "ThetaBetaGauge";
export default ThetaBetaGauge;
