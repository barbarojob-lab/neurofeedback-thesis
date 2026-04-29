/**
 * TranceDepth.tsx  —  frontend/src/components/
 *
 * Barra vertical SVG que representa la profundidad de trance del paciente
 * en unidades de z-score normalizado (−3σ a +3σ).
 *
 * ── Interpretación clínica ────────────────────────────────────────────────
 *
 *   z < −2 : Muy por debajo del baseline → estado de alerta/ansiedad
 *   z ∈ [−2, −1] : Ligeramente activado → arousal elevado
 *   z ∈ [−1,  0] : Bajo baseline propio → estado normal-bajo
 *   z ∈ [ 0, +1] : Sobre baseline → inicio de relajación
 *   z ∈ [+1, +2] : Trance ligero/moderado (zona de feedback positivo)
 *   z > +2 : Trance profundo / estado hipnagógico
 *
 * ── Geometría ────────────────────────────────────────────────────────────
 *
 *   Barra vertical de HEIGHT_BAR px de alto.
 *   El centro (z=0) está a HEIGHT_BAR/2 del top.
 *   Valores positivos crecen HACIA ARRIBA (más relajado = barra más alta).
 *   rango: z ∈ [−3, +3]
 *
 *   y_pixel(z) = BAR_TOP + (1 − (z + 3) / 6) × HEIGHT_BAR
 *   altura_activa(z) = ((z + 3) / 6) × HEIGHT_BAR   (desde abajo)
 */

import { memo, useMemo } from "react";
import { useEEGStore, selectZScore } from "../store/eegStore";

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

const SVG_W      = 100;
const SVG_H      = 280;
const BAR_X      = 44;       // posición X del lado izquierdo de la barra
const BAR_W      = 20;       // ancho de la barra en px
const BAR_TOP    = 20;       // margen superior
const BAR_BOTTOM = 260;      // margen inferior
const BAR_H      = BAR_BOTTOM - BAR_TOP;  // altura total de la barra (240px)
const Z_MIN      = -3;
const Z_MAX      = +3;
const Z_RANGE    = Z_MAX - Z_MIN;         // 6

// Marcas del eje
const Z_MARKS    = [-2, -1, 0, 1, 2];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convierte z-score → posición Y en el SVG (z alto = Y bajo = arriba) */
function zToY(z: number): number {
  const clamped = Math.max(Z_MIN, Math.min(z, Z_MAX));
  return BAR_TOP + (1 - (clamped - Z_MIN) / Z_RANGE) * BAR_H;
}

/** Interpolación de color: rojo (negativo) → gris (0) → verde → cian (positivo) */
function zToColor(z: number): string {
  if (z < 0) {
    // z ∈ [−3, 0]: rojo#ff1744 → gris#607080
    const t = Math.max(0, (z - Z_MIN) / (0 - Z_MIN)); // 0 en z=-3, 1 en z=0
    const r = Math.round(255 + t * (96  - 255));
    const g = Math.round(23  + t * (112 - 23 ));
    const b = Math.round(68  + t * (128 - 68 ));
    return `rgb(${r},${g},${b})`;
  } else {
    // z ∈ [0, +3]: gris#607080 → verde#00e676 → cian#00e5ff
    const t = Math.min(z / Z_MAX, 1); // 0 en z=0, 1 en z=+3
    if (t < 0.5) {
      // gris → verde
      const s = t / 0.5;
      const r = Math.round(96  + s * (0   - 96 ));
      const g = Math.round(112 + s * (230 - 112));
      const b = Math.round(128 + s * (118 - 128));
      return `rgb(${r},${g},${b})`;
    } else {
      // verde → cian
      const s = (t - 0.5) / 0.5;
      const r = Math.round(0);
      const g = Math.round(230 + s * (229 - 230));
      const b = Math.round(118 + s * (255 - 118));
      return `rgb(${r},${g},${b})`;
    }
  }
}

// ---------------------------------------------------------------------------
// Componente
// ---------------------------------------------------------------------------

const TranceDepth = memo(function TranceDepth() {
  const z = useEEGStore(selectZScore);

  const barColor  = useMemo(() => zToColor(z), [z]);
  const yZero     = zToY(0);                    // posición Y de z=0 (centro)
  const yValue    = zToY(z);                    // posición Y del z-score actual

  // La barra activa va desde z=0 hasta z=actual (puede ser hacia arriba o abajo)
  const barTop    = z >= 0 ? yValue : yZero;   // rect.y = el más alto
  const barHeight = Math.abs(yZero - yValue);   // altura absoluta

  // Etiqueta del estado actual
  const stateLabel = useMemo(() => {
    if (z <= -2)      return "MUY ACTIVO";
    if (z <= -1)      return "ACTIVADO";
    if (z <   0)      return "BAJO BASE";
    if (z <   1)      return "RELAJANDO";
    if (z <   2)      return "TRANCE";
    return                   "PROFUNDO";
  }, [z]);

  const clampedZ = Math.max(Z_MIN, Math.min(z, Z_MAX));

  return (
    <div style={{
      display       : "flex",
      flexDirection : "column",
      alignItems    : "center",
      userSelect    : "none",
    }}>
      {/* Etiqueta vertical rotada */}
      <div style={{
        writingMode  : "vertical-rl",
        transform    : "rotate(180deg)",
        fontSize     : 8,
        letterSpacing: "0.15em",
        color        : "rgba(255,255,255,0.3)",
        fontFamily   : "'JetBrains Mono', monospace",
        marginBottom : 6,
        height       : 140,
        display      : "flex",
        alignItems   : "center",
      }}>
        PROFUNDIDAD DE TRANCE
      </div>

      <svg
        width={SVG_W}
        height={SVG_H}
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        aria-label={`Profundidad de trance: z=${z.toFixed(2)}`}
      >
        <defs>
          <filter id="glow-trance">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Gradiente lineal vertical para la barra activa */}
          <linearGradient id="bar-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={barColor} stopOpacity={1} />
            <stop offset="100%" stopColor={barColor} stopOpacity={0.4} />
          </linearGradient>
        </defs>

        {/* ── Barra de fondo (canal completo) ──────────────────────────── */}
        <rect
          x={BAR_X}
          y={BAR_TOP}
          width={BAR_W}
          height={BAR_H}
          rx={BAR_W / 2}
          fill="rgba(255,255,255,0.05)"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={1}
        />

        {/* ── Zona de trance positivo [z=0, z=+3] ligeramente resaltada ── */}
        <rect
          x={BAR_X}
          y={BAR_TOP}
          width={BAR_W}
          height={BAR_H / 2}
          rx={BAR_W / 2}
          fill="rgba(0,230,118,0.06)"
        />

        {/* ── Barra activa (animada con CSS transition) ─────────────────── */}
        <rect
          x={BAR_X}
          y={barTop}
          width={BAR_W}
          height={Math.max(barHeight, 2)} // mínimo 2px visible
          rx={4}
          fill="url(#bar-grad)"
          filter="url(#glow-trance)"
          style={{
            transition: "y 0.35s ease, height 0.35s ease, fill 0.35s ease",
          }}
        />

        {/* ── Línea de baseline (z=0) ───────────────────────────────────── */}
        <line
          x1={BAR_X - 4}
          y1={yZero}
          x2={BAR_X + BAR_W + 4}
          y2={yZero}
          stroke="rgba(255,255,255,0.4)"
          strokeWidth={1}
          strokeDasharray="3 3"
        />

        {/* ── Marcas del eje en −2σ, −1σ, 0, +1σ, +2σ ─────────────────── */}
        {Z_MARKS.map((mark) => {
          const y      = zToY(mark);
          const isZero = mark === 0;
          return (
            <g key={mark}>
              {/* Tick a la izquierda de la barra */}
              <line
                x1={BAR_X - 8}
                y1={y}
                x2={BAR_X - 2}
                y2={y}
                stroke={isZero ? "rgba(255,255,255,0.5)" : "rgba(255,255,255,0.2)"}
                strokeWidth={isZero ? 1.5 : 1}
              />
              {/* Etiqueta */}
              <text
                x={BAR_X - 10}
                y={y + 4}
                textAnchor="end"
                fill={isZero ? "rgba(255,255,255,0.5)" : "rgba(255,255,255,0.25)"}
                fontSize={8}
                fontFamily="'JetBrains Mono', monospace"
              >
                {mark > 0 ? `+${mark}σ` : `${mark}σ`}
              </text>
            </g>
          );
        })}

        {/* ── Indicador de posición actual (línea horizontal coloreada) ─── */}
        <line
          x1={BAR_X - 2}
          y1={zToY(clampedZ)}
          x2={BAR_X + BAR_W + 2}
          y2={zToY(clampedZ)}
          stroke={barColor}
          strokeWidth={2}
          strokeLinecap="round"
          style={{ transition: "y1 0.35s ease, y2 0.35s ease, stroke 0.35s ease" }}
        />

        {/* ── Valor numérico del z-score ────────────────────────────────── */}
        <text
          x={BAR_X + BAR_W / 2}
          y={SVG_H - 6}
          textAnchor="middle"
          fill={barColor}
          fontSize={13}
          fontWeight={700}
          fontFamily="'JetBrains Mono', monospace"
          style={{ transition: "fill 0.35s ease" }}
        >
          {z >= 0 ? "+" : ""}{z.toFixed(2)}σ
        </text>
      </svg>

      {/* Estado textual */}
      <div style={{
        marginTop  : -2,
        fontSize   : 9,
        letterSpacing: "0.14em",
        color      : barColor,
        fontFamily : "'JetBrains Mono', monospace",
        fontWeight : 600,
        transition : "color 0.35s ease",
      }}>
        {stateLabel}
      </div>
    </div>
  );
});

TranceDepth.displayName = "TranceDepth";
export default TranceDepth;
