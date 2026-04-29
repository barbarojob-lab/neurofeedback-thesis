/**
 * ProbabilityMatrix.tsx
 * 
 * Muestra las probabilidades de cada clase (awake, induction, trance)
 * como barras horizontales apiladas.
 */

export interface ProbabilityMatrixProps {
  awake: number;       // 0–1
  induction: number;   // 0–1
  trance: number;      // 0–1
}

export default function ProbabilityMatrix({
  awake,
  induction,
  trance,
}: ProbabilityMatrixProps) {
  // Normalizar para que sumen ~100% (en caso de redondeo numérico)
  const total = awake + induction + trance || 1;
  const normalizedAwake = (awake / total) * 100;
  const normalizedInduction = (induction / total) * 100;
  const normalizedTrance = (trance / total) * 100;

  const classes = [
    {
      name: "Vigilia",
      emoji: "👀",
      color: "#ff5252",
      prob: normalizedAwake,
    },
    {
      name: "Inducción",
      emoji: "🌊",
      color: "#ffc400",
      prob: normalizedInduction,
    },
    {
      name: "Trance",
      emoji: "🧠",
      color: "#00e676",
      prob: normalizedTrance,
    },
  ];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 16,
        padding: "20px",
        borderRadius: 8,
        background: "rgba(13, 13, 26, 0.8)",
        border: "2px solid rgba(255,255,255,0.06)",
        boxShadow: "0 0 12px rgba(0,0,0,0.3), inset 0 0 1px rgba(255,255,255,0.1)",
      }}
    >
      {/* Título */}
      <div
        style={{
          fontSize: "11px",
          fontFamily: "'JetBrains Mono', monospace",
          color: "rgba(255,255,255,0.5)",
          letterSpacing: "0.08em",
          fontWeight: 600,
          textTransform: "uppercase",
        }}
      >
        Classification Probabilities
      </div>

      {/* Barras de probabilidad */}
      {classes.map((cls) => (
        <div key={cls.name}>
          {/* Label con emoji */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: "4px",
            }}
          >
            <span style={{ fontSize: "16px" }}>{cls.emoji}</span>
            <span
              style={{
                fontSize: "10px",
                fontFamily: "'JetBrains Mono', monospace",
                color: cls.color,
                fontWeight: 600,
                width: "90px",
              }}
            >
              {cls.name}
            </span>
            <span
              style={{
                fontSize: "12px",
                fontWeight: 700,
                color: cls.color,
                fontFamily: "'JetBrains Mono', monospace",
                letterSpacing: "0.05em",
                minWidth: "45px",
                textAlign: "right",
              }}
            >
              {cls.prob.toFixed(1)}%
            </span>
          </div>

          {/* Barra horizontal */}
          <div
            style={{
              height: "24px",
              background: "rgba(0,0,0,0.3)",
              borderRadius: "4px",
              overflow: "hidden",
              border: `1px solid ${cls.color}33`,
              position: "relative",
            }}
          >
            {/* Relleno proporcionalmente */}
            <div
              style={{
                height: "100%",
                width: `${cls.prob}%`,
                background: `linear-gradient(90deg, ${cls.color}22 0%, ${cls.color}55 100%)`,
                borderRadius: "3px",
                transition: "width 0.4s ease",
                boxShadow: `inset 0 0 6px ${cls.color}44, 0 0 8px ${cls.color}22`,
              }}
            />

            {/* Overlay de líneas para efecto 3D */}
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: "1px",
                background: `linear-gradient(90deg, ${cls.color}22 0%, transparent 100%)`,
              }}
            />
          </div>
        </div>
      ))}

      {/* Footer — info */}
      <div
        style={{
          fontSize: "9px",
          color: "rgba(255,255,255,0.3)",
          fontFamily: "'JetBrains Mono', monospace",
          marginTop: "8px",
          borderTop: "1px solid rgba(255,255,255,0.06)",
          paddingTop: "8px",
          textAlign: "center",
          letterSpacing: "0.05em",
        }}
      >
        Sum: {normalizedAwake.toFixed(1)} + {normalizedInduction.toFixed(1)} +{" "}
        {normalizedTrance.toFixed(1)} = {(normalizedAwake + normalizedInduction + normalizedTrance).toFixed(1)}%
      </div>
    </div>
  );
}
