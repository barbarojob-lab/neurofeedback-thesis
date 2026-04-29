/**
 * ConfidenceGauge.tsx
 * 
 * Visualiza la confianza de la predicción del clasificador (0–100%).
 * Usa indicador de aguja + color dinámico.
 */

export interface ConfidenceGaugeProps {
  confidence: number;  // 0–1
}

export default function ConfidenceGauge({ confidence }: ConfidenceGaugeProps) {
  const percent = confidence * 100;
  
  // Color dinámico según confianza
  let color: string;
  let status: string;
  
  if (percent >= 80) {
    color = "#00e676";  // Verde: muy confiado
    status = "HIGH";
  } else if (percent >= 60) {
    color = "#ffc400";  // Amarillo: moderado
    status = "MEDIUM";
  } else if (percent >= 40) {
    color = "#ff9800";  // Naranja: bajo
    status = "LOW";
  } else {
    color = "#ff5252";  // Rojo: muy bajo
    status = "VERY LOW";
  }

  const rotation = (percent / 100) * 180 - 90; // Mapear 0–100 a -90° a 90°

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 12,
        padding: "20px",
        borderRadius: 8,
        background: "rgba(13, 13, 26, 0.8)",
        border: `2px solid ${color}66`,
        boxShadow: `0 0 12px ${color}22, inset 0 0 1px rgba(255,255,255,0.1)`,
        transition: "all 0.3s ease",
        minHeight: "200px",
      }}
    >
      {/* Gauge visual — SVG semicírculo */}
      <svg
        width="120"
        height="80"
        viewBox="0 0 120 80"
        style={{ overflow: "visible" }}
      >
        {/* Fondo de la escala */}
        <path
          d="M 10 70 A 60 60 0 0 1 110 70"
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="4"
          strokeLinecap="round"
        />

        {/* Zona verde (80–100%) */}
        <path
          d="M 95 70 A 60 60 0 0 1 110 70"
          fill="none"
          stroke="rgba(0,230,118,0.3)"
          strokeWidth="4"
          strokeLinecap="round"
        />

        {/* Zona amarilla (60–80%) */}
        <path
          d="M 75 70 A 60 60 0 0 1 95 70"
          fill="none"
          stroke="rgba(255,196,0,0.3)"
          strokeWidth="4"
          strokeLinecap="round"
        />

        {/* Zona naranja (40–60%) */}
        <path
          d="M 55 70 A 60 60 0 0 1 75 70"
          fill="none"
          stroke="rgba(255,152,0,0.3)"
          strokeWidth="4"
          strokeLinecap="round"
        />

        {/* Zona roja (0–40%) */}
        <path
          d="M 10 70 A 60 60 0 0 1 55 70"
          fill="none"
          stroke="rgba(255,82,82,0.3)"
          strokeWidth="4"
          strokeLinecap="round"
        />

        {/* Aguja indicadora */}
        <g transform={`rotate(${rotation} 60 70)`}>
          <line
            x1="60"
            y1="70"
            x2="60"
            y2="20"
            stroke={color}
            strokeWidth="3"
            strokeLinecap="round"
            style={{
              filter: `drop-shadow(0 0 4px ${color})`,
              transition: "all 0.3s ease",
            }}
          />
          {/* Punta redonda de la aguja */}
          <circle cx="60" cy="20" r="4" fill={color} />
        </g>

        {/* Marcas de escala (0, 50, 100) */}
        <text
          x="10"
          y="75"
          fontSize="9"
          fill="rgba(255,255,255,0.3)"
          fontFamily="'JetBrains Mono', monospace"
          textAnchor="start"
        >
          0%
        </text>
        <text
          x="60"
          y="85"
          fontSize="9"
          fill="rgba(255,255,255,0.3)"
          fontFamily="'JetBrains Mono', monospace"
          textAnchor="middle"
        >
          50%
        </text>
        <text
          x="110"
          y="75"
          fontSize="9"
          fill="rgba(255,255,255,0.3)"
          fontFamily="'JetBrains Mono', monospace"
          textAnchor="end"
        >
          100%
        </text>

        {/* Centro circle */}
        <circle cx="60" cy="70" r="5" fill="rgba(0,0,0,0.3)" />
      </svg>

      {/* Valor numérico grande */}
      <div
        style={{
          fontSize: "28px",
          fontWeight: 700,
          color,
          fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: "0.08em",
          textShadow: `0 0 8px ${color}33`,
        }}
      >
        {percent.toFixed(0)}%
      </div>

      {/* Status label */}
      <div
        style={{
          fontSize: "10px",
          color,
          fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: "0.1em",
          fontWeight: 600,
        }}
      >
        {status} CONFIDENCE
      </div>

      {/* Interpretación */}
      <div
        style={{
          fontSize: "9px",
          color: "rgba(255,255,255,0.5)",
          fontFamily: "'JetBrains Mono', monospace",
          textAlign: "center",
          marginTop: "4px",
        }}
      >
        {percent >= 70
          ? "Model prediction is reliable"
          : "Collecting more data for validation"}
      </div>
    </div>
  );
}
