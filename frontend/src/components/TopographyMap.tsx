import { memo, useMemo } from "react";
import { useEEGStore, selectTopographyTheta } from "../store/eegStore";

const ELECTRODE_POSITIONS: Array<{ name: string; x: number; y: number; primary?: boolean }> = [
  { name: "Fp1", x: 30, y: 22 },
  { name: "Fp2", x: 70, y: 22 },
  { name: "F3", x: 34, y: 38 },
  { name: "Fz", x: 50, y: 34, primary: true },
  { name: "F4", x: 66, y: 38 },
  { name: "C3", x: 34, y: 55 },
  { name: "Cz", x: 50, y: 52 },
  { name: "C4", x: 66, y: 55 },
  { name: "P3", x: 36, y: 70 },
  { name: "Pz", x: 50, y: 72 },
  { name: "P4", x: 64, y: 70 },
  { name: "O1", x: 40, y: 84 },
  { name: "O2", x: 60, y: 84 },
];

function heatColor(v: number): string {
  const t = Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
  const r = Math.round(18 + 235 * t);
  const g = Math.round(42 + 168 * (1 - Math.abs(t - 0.5) * 1.9));
  const b = Math.round(255 - 235 * t);
  return `rgb(${r},${g},${b})`;
}

function interpolateIDW(
  x: number,
  y: number,
  points: Array<{ x: number; y: number; value: number }>
): number {
  let weightedSum = 0;
  let weightSum = 0;

  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const dx = x - p.x;
    const dy = y - p.y;
    const d2 = dx * dx + dy * dy;

    // Si estamos prácticamente encima de un electrodo, usar valor directo.
    if (d2 < 0.5) return p.value;

    const w = 1 / Math.pow(d2, 1.15);
    weightedSum += p.value * w;
    weightSum += w;
  }

  if (weightSum <= 1e-12) return 0;
  return Math.max(0, Math.min(1, weightedSum / weightSum));
}

const TopographyMap = memo(function TopographyMap() {
  const topography = useEEGStore(selectTopographyTheta);

  const points = useMemo(
    () => ELECTRODE_POSITIONS.map((e) => ({ ...e, value: topography[e.name] ?? 0 })),
    [topography]
  );

  const heatCells = useMemo(() => {
    const cells: Array<{ x: number; y: number; color: string; opacity: number }> = [];
    const minX = 16;
    const maxX = 84;
    const minY = 12;
    const maxY = 92;
    const step = 2;
    const cx = 50;
    const cy = 52;
    const rx = 34;
    const ry = 40;

    for (let y = minY; y <= maxY; y += step) {
      for (let x = minX; x <= maxX; x += step) {
        // Mantener solo puntos dentro de la elipse craneal.
        const ex = (x - cx) / rx;
        const ey = (y - cy) / ry;
        const inside = ex * ex + ey * ey <= 1;
        if (!inside) continue;

        const v = interpolateIDW(x, y, points);
        // Borde más tenue para que el centro tenga más protagonismo.
        const radial = Math.max(0, 1 - (ex * ex + ey * ey));
        const opacity = 0.2 + radial * 0.65;
        cells.push({ x, y, color: heatColor(v), opacity });
      }
    }

    return cells;
  }, [points]);

  return (
    <div style={{ width: "100%" }}>
      <div style={{
        fontSize: 9,
        fontFamily: "'JetBrains Mono', monospace",
        letterSpacing: "0.12em",
        color: "rgba(255,255,255,0.3)",
        marginBottom: 8,
        textAlign: "center",
      }}>
        TOPOGRAFIA THETA
      </div>

      <svg viewBox="0 0 100 100" width="100%" height="220" role="img" aria-label="Topografia theta">
        <defs>
          <clipPath id="scalp-clip">
            <ellipse cx="50" cy="52" rx="34" ry="40" />
          </clipPath>
          <filter id="scalp-blur">
            <feGaussianBlur stdDeviation="1.2" />
          </filter>
        </defs>

        <rect x="0" y="0" width="100" height="100" fill="rgba(255,255,255,0.01)" />

        <g clipPath="url(#scalp-clip)" filter="url(#scalp-blur)">
          {heatCells.map((c, idx) => (
            <rect
              key={idx}
              x={c.x - 1.2}
              y={c.y - 1.2}
              width={2.4}
              height={2.4}
              fill={c.color}
              opacity={c.opacity}
            />
          ))}
        </g>

        <ellipse cx="50" cy="52" rx="34" ry="40" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="1" />
        <path d="M 46 12 L 50 8 L 54 12" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="1" />

        {points.map((p) => (
          <g key={p.name}>
            <circle
              cx={p.x}
              cy={p.y}
              r={p.primary ? 4.8 : 4}
              fill={heatColor(p.value)}
              stroke={p.primary ? "#ffffff" : "rgba(255,255,255,0.4)"}
              strokeWidth={p.primary ? 1.3 : 0.8}
            />
            <text
              x={p.x}
              y={p.y + (p.primary ? 8 : 7)}
              textAnchor="middle"
              fontSize="3.2"
              fill={p.primary ? "#ffffff" : "rgba(255,255,255,0.75)"}
              fontFamily="'JetBrains Mono', monospace"
            >
              {p.name}
            </text>
          </g>
        ))}
      </svg>

      <div style={{
        display: "flex",
        justifyContent: "space-between",
        fontSize: 8,
        color: "rgba(255,255,255,0.35)",
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        <span>0.0</span>
        <span style={{ color: "#8ab4ff" }}>azul</span>
        <span style={{ color: "#ff6b6b" }}>rojo</span>
        <span>1.0</span>
      </div>
    </div>
  );
});

export default TopographyMap;
