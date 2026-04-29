/**
 * StateClassifier.tsx
 * 
 * Muestra el estado predicho actual del paciente (awake, induction, trance)
 * basado en la clasificación del modelo ML en tiempo real.
 */

import { useEffect, useState } from "react";

export interface StateClassifierProps {
  label: "awake" | "induction" | "trance" | null;
  confidence: number;  // 0–1
  timestamp?: number;  // cuando se hizo la predicción
}

export default function StateClassifier({
  label,
  confidence,
  timestamp,
}: StateClassifierProps) {
  const isConfident = confidence >= 0.70;
  const [predictedAgoSec, setPredictedAgoSec] = useState(0);

  // Actualiza cada segundo para mostrar "hace X segundos"
  useEffect(() => {
    if (!timestamp) return;
    const update = () =>
      setPredictedAgoSec(Math.floor((Date.now() - timestamp) / 1000));
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, [timestamp]);

  const stateInfo: Record<
    "awake" | "induction" | "trance" | "calibrating",
    { emoji: string; label: string; color: string; description: string }
  > = {
    awake: {
      emoji: "👀",
      label: "AWAKE",
      color: "#ff5252",
      description: "Alert state — beta dominant",
    },
    induction: {
      emoji: "🌊",
      label: "INDUCTION",
      color: "#ffc400",
      description: "Transitioning to trance",
    },
    trance: {
      emoji: "🧠",
      label: "TRANCE",
      color: "#00e676",
      description: "Deep theta activity detected",
    },
    calibrating: {
      emoji: "⏳",
      label: "CALIBRATING",
      color: "rgba(255,255,255,0.5)",
      description: "Warm-up period",
    },
  };

  const display =
    label && isConfident ? stateInfo[label] : stateInfo.calibrating;

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
        border: `2px solid ${display.color}66`,
        boxShadow: `0 0 12px ${display.color}22, inset 0 0 1px rgba(255,255,255,0.1)`,
        transition: "all 0.3s ease",
      }}
    >
      {/* Emoji grande */}
      <div
        style={{
          fontSize: "48px",
          lineHeight: "1",
          height: "52px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {display.emoji}
      </div>

      {/* Etiqueta de estado */}
      <div
        style={{
          fontSize: "16px",
          fontWeight: 700,
          letterSpacing: "0.12em",
          color: display.color,
          fontFamily: "'JetBrains Mono', monospace",
          textAlign: "center",
          textShadow: `0 0 8px ${display.color}33`,
        }}
      >
        {display.label}
      </div>

      {/* Descripción */}
      <div
        style={{
          fontSize: "11px",
          color: "rgba(255,255,255,0.5)",
          textAlign: "center",
          fontFamily: "'JetBrains Mono', monospace",
          letterSpacing: "0.05em",
        }}
      >
        {display.description}
      </div>

      {/* Metadata — timestamp y confianza */}
      <div
        style={{
          fontSize: "9px",
          color: "rgba(255,255,255,0.35)",
          fontFamily: "'JetBrains Mono', monospace",
          textAlign: "center",
          marginTop: "4px",
        }}
      >
        Predicted {predictedAgoSec}s ago
      </div>

      {/* Status indicator — confident vs calibrating */}
      <div
        style={{
          fontSize: "10px",
          fontFamily: "'JetBrains Mono', monospace",
          color: isConfident ? "#00e676" : "rgba(255,255,255,0.3)",
          letterSpacing: "0.08em",
        }}
      >
        {isConfident ? "✓ CONFIDENT" : "◆ LOW CONFIDENCE"}
      </div>
    </div>
  );
}
