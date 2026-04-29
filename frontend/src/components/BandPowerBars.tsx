/**
 * BandPowerBars.tsx  —  frontend/src/components/
 *
 * 5 barras horizontales que muestran la potencia espectral relativa de cada
 * banda EEG: δ, θ, α, β, γ.
 *
 * ── Normalización dinámica ────────────────────────────────────────────────
 *
 *   Cada barra se normaliza al 100% del MÁXIMO visto en la sesión para esa
 *   banda. Esto hace que la visualización sea "relativa al pico de sesión",
 *   lo que permite comparar qué bandas están activas entre sí, independientemente
 *   de la escala absoluta del paciente.
 *
 *   maxRef[banda] = max(maxRef[banda], valor_actual)
 *
 *   El maxRef se guarda en un useRef (no en estado) para no causar re-renders
 *   adicionales cuando se actualiza.
 *
 * ── Frecuencia de actualización ──────────────────────────────────────────
 *
 *   bandPowers se actualiza en el store ~4 veces/segundo (cada hopSize=64
 *   muestras a 250 Hz = 256 ms). El componente se suscribe al selector
 *   selectBandPowers y re-renderiza SOLO cuando cambia ese slice del estado.
 *   Con 5 barras CSS simples, el render es < 0.5 ms.
 *
 * ── Color especial de theta ──────────────────────────────────────────────
 *
 *   Theta (#ffd600 dorado/ámbar) se distingue visualmente de las demás:
 *   - Es la señal objetivo del neurofeedback de relajación.
 *   - El terapeuta necesita verla destacada en cualquier condición.
 *   - Las demás bandas usan colores suaves para no competir con theta.
 */

import { memo, useMemo } from "react";
import {
  useEEGStore,
  selectBandPowers,
  selectSessionMaxBandPowers,
} from "../store/eegStore";
import type { BandPowers } from "../types";

// ---------------------------------------------------------------------------
// Configuración de las bandas
// ---------------------------------------------------------------------------

interface BandConfig {
  key    : keyof Omit<BandPowers, "timestamp">;
  label  : string;   // símbolo de la banda
  name   : string;   // nombre completo para tooltip / accesibilidad
  range  : string;   // rango de frecuencias
  color  : string;   // color de la barra
  glow   : string;   // color de la sombra de brillo
}

const BAND_CONFIG: BandConfig[] = [
  {
    key  : "delta",
    label: "δ  Delta",
    name : "Delta",
    range: "1–4 Hz",
    color: "#5c6bc0",   // índigo apagado — sueño profundo / DC
    glow : "rgba(92,107,192,0.4)",
  },
  {
    key  : "theta",
    label: "θ  Theta",
    name : "Theta",
    range: "4–8 Hz",
    color: "#ffd600",   // dorado/ámbar — señal principal de trance
    glow : "rgba(255,214,0,0.5)",
  },
  {
    key  : "alpha",
    label: "α  Alpha",
    name : "Alpha",
    range: "8–12 Hz",
    color: "#26c6da",   // cian suave — relajación alerta
    glow : "rgba(38,198,218,0.35)",
  },
  {
    key  : "beta",
    label: "β  Beta",
    name : "Beta",
    range: "12–30 Hz",
    color: "#ef5350",   // rojo suave — activación cognitiva
    glow : "rgba(239,83,80,0.35)",
  },
  {
    key  : "gamma",
    label: "γ  Gamma",
    name : "Gamma",
    range: "30–45 Hz",
    color: "rgba(255,255,255,0.25)", // gris translúcido — offline / EMG
    glow : "none",
  },
];

// Valor mínimo para evitar log(0) si las bandas están a cero (inicio de sesión)
const MIN_POWER = 1e-6;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Devuelve el porcentaje de la barra [0, 100] para un valor dado su máximo */
function toPercent(value: number, max: number): number {
  if (max < MIN_POWER) return 0;
  return Math.min(100, (value / max) * 100);
}

// ---------------------------------------------------------------------------
// Subcomponente de barra individual
// ---------------------------------------------------------------------------

interface BandBarProps {
  config  : BandConfig;
  percent : number;   // [0, 100]
  rawValue: number;   // µV² para mostrar en tooltip
  isTheta : boolean;
}

const BandBar = memo(function BandBar({
  config,
  percent,
  rawValue,
  isTheta,
}: BandBarProps) {
  return (
    <div
      title={`${config.name} (${config.range}): ${rawValue.toFixed(4)} µV²`}
      style={{
        display       : "flex",
        alignItems    : "center",
        gap           : 8,
        marginBottom  : isTheta ? 10 : 7,
      }}
    >
      {/* Etiqueta de banda */}
      <div style={{
        width      : 70,
        fontSize   : isTheta ? 11 : 10,
        fontFamily : "'JetBrains Mono', monospace",
        color      : isTheta ? config.color : "rgba(255,255,255,0.45)",
        fontWeight : isTheta ? 700 : 400,
        letterSpacing: "0.04em",
        flexShrink : 0,
        textAlign  : "right",
      }}>
        {config.label}
      </div>

      {/* Barra de progreso */}
      <div style={{
        flex        : 1,
        height      : isTheta ? 14 : 10,
        backgroundColor: "rgba(255,255,255,0.05)",
        borderRadius: 3,
        overflow    : "hidden",
        position    : "relative",
        border      : isTheta
          ? `1px solid rgba(255,214,0,0.25)`
          : "1px solid rgba(255,255,255,0.06)",
      }}>
        <div style={{
          height      : "100%",
          width       : `${percent}%`,
          backgroundColor: config.color,
          borderRadius: 3,
          boxShadow   : config.glow !== "none" && percent > 5
            ? `0 0 8px ${config.glow}`
            : "none",
          // Transición CSS — la barra se anima suavemente cada 256 ms
          transition  : "width 0.25s ease, box-shadow 0.25s ease",
        }} />
      </div>

      {/* Porcentaje */}
      <div style={{
        width      : 38,
        fontSize   : 9,
        fontFamily : "'JetBrains Mono', monospace",
        color      : isTheta ? config.color : "rgba(255,255,255,0.25)",
        textAlign  : "right",
        flexShrink : 0,
      }}>
        {percent.toFixed(0)}%
      </div>
    </div>
  );
});

// ---------------------------------------------------------------------------
// Componente principal
// ---------------------------------------------------------------------------

const BandPowerBars = memo(function BandPowerBars() {
  const bandPowers = useEEGStore(selectBandPowers);
  const sessionMax = useEEGStore(selectSessionMaxBandPowers);

  // Calcular porcentajes en O(5)
  const barData = useMemo(() => {
    if (!bandPowers) {
      return BAND_CONFIG.map((c) => ({ config: c, percent: 0, rawValue: 0 }));
    }

    return BAND_CONFIG.map((cfg) => {
      const v       = bandPowers[cfg.key] ?? 0;
      const max     = sessionMax[cfg.key];
      const percent = toPercent(v, max);
      return { config: cfg, percent, rawValue: v };
    });
  }, [bandPowers, sessionMax]);

  return (
    <div style={{
      width    : "100%",
      padding  : "4px 0",
    }}>
      {/* Título */}
      <div style={{
        fontSize     : 9,
        fontFamily   : "'JetBrains Mono', monospace",
        letterSpacing: "0.14em",
        color        : "rgba(255,255,255,0.25)",
        marginBottom : 14,
        textAlign    : "center",
      }}>
        POTENCIA ESPECTRAL  (%  sesión max)
      </div>

      {/* Barras */}
      {barData.map(({ config, percent, rawValue }) => (
        <BandBar
          key={config.key}
          config={config}
          percent={percent}
          rawValue={rawValue}
          isTheta={config.key === "theta"}
        />
      ))}

      {/* Advertencia gamma */}
      <div style={{
        fontSize     : 8,
        fontFamily   : "'JetBrains Mono', monospace",
        color        : "rgba(255,255,255,0.18)",
        letterSpacing: "0.06em",
        marginTop    : 6,
        textAlign    : "right",
      }}>
        ⚠ γ solo offline (EMG)
      </div>
    </div>
  );
});

BandPowerBars.displayName = "BandPowerBars";
export default BandPowerBars;
