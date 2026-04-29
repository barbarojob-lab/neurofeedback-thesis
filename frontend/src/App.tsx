/**
 * App.tsx  —  frontend/src/
 *
 * Dashboard principal del sistema de neurofeedback EEG.
 *
 * ── Layout CSS Grid ───────────────────────────────────────────────────────
 *
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │  HEADER — título + indicador WS + info de sesión                   │
 *   ├────────────────────────────────────────────────────────────────────┤
 *   │  WAVEFORM  (col-span: completo)                                    │
 *   │  WaveformChart — osciloscopio EEG en tiempo real                   │
 *   ├──────────────────┬────────────────┬──────────────┬─────────────────┤
 *   │  ThetaBetaGauge  │  TranceDepth   │  BandPower   │  PANEL CONTROL  │
 *   │  (ratio Θ/β)     │  (z-score)     │  (5 bandas)  │  start/stop     │
 *   │                  │                │              │  paciente / grupo│
 *   └──────────────────┴────────────────┴──────────────┴─────────────────┘
 *
 * ── Paleta de color ──────────────────────────────────────────────────────
 *
 *   #080810 — fondo base (negro azulado)
 *   #0d0d1a — fondo de paneles
 *   #111128 — fondo de inputs / controles
 *   #00e5ff — acento cian (conectado, activo)
 *   #ff1744 — alerta / desconectado
 *   rgba(255,255,255,0.06) — bordes de paneles
 */

import React, {
  useState,
  useCallback,
  useId,
} from "react";

import WaveformChart    from "./components/WaveformChart";
import ThetaBetaGauge   from "./components/ThetaBetaGauge";
import BandPowerBars    from "./components/BandPowerBars";
import TopographyMap    from "./components/TopographyMap";
import ChannelInspectionPanel from "./components/ChannelInspectionPanel";
// ✅ NUEVOS: Componentes del clasificador
import StateClassifier  from "./components/StateClassifier";
import ConfidenceGauge  from "./components/ConfidenceGauge";
import ProbabilityMatrix from "./components/ProbabilityMatrix";
import { useEEGSocket } from "./hooks/useEEGSocket";
import {
  useEEGStore,
  selectPipelineMs,
  selectLastError,
  selectFrontalSpecificity,
  selectFrontalSpecificityValid,
  selectArtifactDetected,
  selectStatePrediction,
} from "./store/eegStore";

import type { SessionConfig, DatasetMetadata } from "./types";

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

const SAMPLE_RATE = 250;
const EEG_CHANNEL_OPTIONS = ["Fz", "Fp1", "Fp2", "F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"];

// ---------------------------------------------------------------------------
// Sub-componentes de UI local
// ---------------------------------------------------------------------------

/** Punto indicador de estado de conexión WS */
function ConnectionDot({
  connected,
  retryAttempt = 0,
}: {
  connected: boolean;
  retryAttempt?: number;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div style={{
        width           : 10,
        height          : 10,
        borderRadius    : "50%",
        backgroundColor : connected ? "#00e676" : "#ff1744",
        boxShadow       : connected
          ? "0 0 8px #00e676, 0 0 16px rgba(0,230,118,0.4)"
          : "0 0 6px #ff1744",
        flexShrink      : 0,
        transition      : "background-color 0.3s, box-shadow 0.3s",
      }} />
      <span style={{
        fontSize     : 10,
        fontFamily   : "'JetBrains Mono', monospace",
        color        : connected ? "#00e676" : "#ff5252",
        letterSpacing: "0.08em",
      }}>
        {connected
          ? "WS CONECTADO"
          : retryAttempt > 0
            ? `WS DESCONECTADO (retry ${retryAttempt})`
            : "WS DESCONECTADO"}
      </span>
    </div>
  );
}

/** Badge dinámico de estado predicho (reemplaza CommandBadge) */
function StatePredictionBadge() {
  const prediction = useEEGStore(selectStatePrediction);
  
  if (!prediction) {
    return (
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "4px 12px",
        borderRadius: 4,
        background: "rgba(96,125,139,0.15)",
        border: "1px solid rgba(144,164,174,0.33)",
      }}>
        <span style={{
          fontSize: 10,
          fontFamily: "'JetBrains Mono', monospace",
          color: "#90a4ae",
          letterSpacing: "0.1em",
          fontWeight: 600,
        }}>
          ⏳ WAITING FOR PREDICTION
        </span>
      </div>
    );
  }

  const stateColors: Record<string, string> = {
    awake: "#ff5252",
    induction: "#ffc400",
    trance: "#00e676",
  };

  const label = prediction.predicted_label.toUpperCase();
  const color = stateColors[prediction.predicted_label] ?? "#90a4ae";

  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: 8,
      padding: "4px 12px",
      borderRadius: 4,
      background: `${color}15`,
      border: `1px solid ${color}33`,
    }}>
      <span style={{
        fontSize: 10,
        fontFamily: "'JetBrains Mono', monospace",
        color: color,
        letterSpacing: "0.1em",
        fontWeight: 600,
      }}>
        {label}
      </span>
      <span style={{
        fontSize: 10,
        color: "rgba(255,255,255,0.3)",
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        {(prediction.confidence * 100).toFixed(0)}%
      </span>
    </div>
  );
}


/** Indicador de latencia del pipeline */
function PipelineLatency() {
  const ms = useEEGStore(selectPipelineMs);
  const color = ms === 0 ? "rgba(255,255,255,0.2)"
    : ms < 5  ? "#00e676"
    : ms < 10 ? "#ffd600"
    : "#ff5252";

  return (
    <span style={{
      fontSize : 9,
      fontFamily: "'JetBrains Mono', monospace",
      color,
    }}>
      DSP {ms === 0 ? "--" : `${ms.toFixed(1)}`} ms
    </span>
  );
}

/** Panel de control de sesión (lado derecho) */
interface ControlPanelProps {
  isConnected       : boolean;
  isSessionActive   : boolean;
  sessionId         : string | null;
  onStart           : (cfg: SessionConfig) => void;
  onStop            : () => void;
  onTranceToggle    : (enabled: boolean) => void;
  onSubmitNRS       : (sessionId: string, value: number) => void;
  onPing            : () => void;
  inspectionChannel : string;
  onInspectionChannelChange: (channel: string) => void;
  onLoadDataset: (path: string) => void;
  dataset: DatasetMetadata | null;
}

function ControlPanel({
  isConnected,
  isSessionActive,
  sessionId,
  onStart,
  onStop,
  onTranceToggle,
  onSubmitNRS,
  onPing,
  inspectionChannel,
  onInspectionChannelChange,
  onLoadDataset,
  dataset,
}: ControlPanelProps) {
  const [patientId,    setPatientId]    = useState("");
  const [group,        setGroup]        = useState<"experimental" | "control">("experimental");
  const [tranceMode,   setTranceModeLocal] = useState(false);
  const [sigmoidK,     setSigmoidK]     = useState(2.0);
  const [thetaThresh,  setThetaThresh]  = useState(0.0);
  const [datasetPath, setDatasetPath] = useState("");

  const patientIdId = useId();
  const groupId     = useId();

  const handleStart = useCallback(() => {
    const config: SessionConfig = {
      thetaThreshold        : thetaThresh,
      thetaPeakTbrThreshold : 2.5,
      sigmoidK,
    };
    onStart(config);
  }, [onStart, thetaThresh, sigmoidK]);

  const handleTranceToggle = useCallback(() => {
    const next = !tranceMode;
    setTranceModeLocal(next);
    onTranceToggle(next);
  }, [tranceMode, onTranceToggle]);

  const inputStyle: React.CSSProperties = {
    width           : "100%",
    padding         : "7px 10px",
    borderRadius    : 4,
    border          : "1px solid rgba(255,255,255,0.12)",
    backgroundColor : "#111128",
    color           : "rgba(255,255,255,0.85)",
    fontSize        : 12,
    fontFamily      : "'JetBrains Mono', monospace",
    outline         : "none",
    boxSizing       : "border-box",
  };

  const labelStyle: React.CSSProperties = {
    fontSize     : 9,
    fontFamily   : "'JetBrains Mono', monospace",
    color        : "rgba(255,255,255,0.35)",
    letterSpacing: "0.1em",
    marginBottom : 4,
    display      : "block",
  };

  const field = (label: string, children: React.ReactNode) => (
    <div style={{ marginBottom: 14 }}>
      <label style={labelStyle}>{label}</label>
      {children}
    </div>
  );

  return (
    <div style={{
      backgroundColor: "#0d0d1a",
      border         : "1px solid rgba(255,255,255,0.07)",
      borderRadius   : 8,
      padding        : 18,
      display        : "flex",
      flexDirection  : "column",
      gap            : 0,
      height         : "100%",
      boxSizing      : "border-box",
    }}>
      <div style={{
        fontSize     : 9,
        fontFamily   : "'JetBrains Mono', monospace",
        color        : "rgba(255,255,255,0.25)",
        letterSpacing: "0.15em",
        marginBottom : 16,
      }}>
        CONTROL DE SESIÓN
      </div>

      {field("ID DE PACIENTE",
        <input
          id={patientIdId}
          type="text"
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          placeholder="P001"
          disabled={isSessionActive}
          style={inputStyle}
        />
      )}

      {field("GRUPO",
        <select
          id={groupId}
          value={group}
          onChange={(e) => setGroup(e.target.value as typeof group)}
          disabled={isSessionActive}
          style={inputStyle}
        >
          <option value="experimental">Experimental (NF activo)</option>
          <option value="control">Control (NF sham)</option>
        </select>
      )}

      {field("SIGMOID K (sensibilidad)",
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="range"
            min={0.5}
            max={5}
            step={0.5}
            value={sigmoidK}
            disabled={isSessionActive}
            onChange={(e) => setSigmoidK(Number(e.target.value))}
            style={{ flex: 1, accentColor: "#00e5ff" }}
          />
          <span style={{
            fontSize  : 11,
            fontFamily: "'JetBrains Mono', monospace",
            color     : "#00e5ff",
            minWidth  : 24,
          }}>
            {sigmoidK}
          </span>
        </div>
      )}

      {field("UMBRAL Θ (z-score)",
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="range"
            min={-1}
            max={1.5}
            step={0.1}
            value={thetaThresh}
            disabled={isSessionActive}
            onChange={(e) => setThetaThresh(Number(e.target.value))}
            style={{ flex: 1, accentColor: "#ffd600" }}
          />
          <span style={{
            fontSize  : 11,
            fontFamily: "'JetBrains Mono', monospace",
            color     : "#ffd600",
            minWidth  : 32,
          }}>
            {thetaThresh >= 0 ? "+" : ""}{thetaThresh.toFixed(1)}σ
          </span>
        </div>
      )}

      {field("CANAL INSPECCION",
        <select
          value={inspectionChannel}
          onChange={(e) => onInspectionChannelChange(e.target.value)}
          style={inputStyle}
        >
          {EEG_CHANNEL_OPTIONS.map((ch) => (
            <option key={ch} value={ch}>{ch}</option>
          ))}
        </select>
      )}

      {field("DATASET (ruta local)",
        <div style={{ display: "grid", gap: 6 }}>
          <input
            type="text"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            placeholder="D:/datasets/eeg/sesion01.edf"
            style={inputStyle}
          />
          <button
            onClick={() => onLoadDataset(datasetPath.trim())}
            disabled={!datasetPath.trim() || !isConnected}
            style={{
              width: "100%",
              padding: "6px 0",
              borderRadius: 5,
              border: "none",
              cursor: !datasetPath.trim() || !isConnected ? "not-allowed" : "pointer",
              backgroundColor: "rgba(0,229,255,0.12)",
              color: "#00e5ff",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10,
              letterSpacing: "0.08em",
              outline: "1px solid rgba(0,229,255,0.25)",
            } as React.CSSProperties}
          >
            CARGAR DATASET
          </button>
        </div>
      )}

      {dataset && (
        <div style={{
          padding: "8px 10px",
          borderRadius: 6,
          marginBottom: 10,
          background: "rgba(0,229,255,0.06)",
          border: "1px solid rgba(0,229,255,0.18)",
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 9,
          color: "rgba(255,255,255,0.75)",
          display: "grid",
          gap: 3,
        }}>
          <div>FMT: {dataset.format.toUpperCase()}</div>
          <div>CH: {dataset.channels.length}</div>
          <div>FS: {dataset.sampleRate} Hz</div>
          <div>DUR: {dataset.durationSec.toFixed(1)} s</div>
        </div>
      )}

      {/* Botón principal: Iniciar / Detener */}
      <button
        onClick={isSessionActive ? onStop : handleStart}
        disabled={!isConnected}
        style={{
          width          : "100%",
          padding        : "10px 0",
          borderRadius   : 5,
          border         : "none",
          cursor         : isConnected ? "pointer" : "not-allowed",
          backgroundColor: isSessionActive
            ? "rgba(255,23,68,0.2)"
            : isConnected
              ? "rgba(0,229,255,0.18)"
              : "rgba(255,255,255,0.06)",
          color          : isSessionActive
            ? "#ff5252"
            : isConnected ? "#00e5ff" : "rgba(255,255,255,0.3)",
          // border_color: isSessionActive ? "#ff524244" : "#00e5ff44", // removed invalid prop
          fontFamily     : "'JetBrains Mono', monospace",
          fontSize       : 12,
          fontWeight     : 700,
          letterSpacing  : "0.12em",
          transition     : "all 0.2s",
          marginTop      : 4,
          outline        : "1px solid",
          borderColor: isSessionActive ? "rgba(255,82,82,0.35)" : "rgba(0,229,255,0.3)",

        } as React.CSSProperties}
      >
        {isSessionActive ? "■  DETENER SESIÓN" : "▶  INICIAR SESIÓN"}
      </button>

      {/* Modo trance — solo disponible con sesión activa */}
      {isSessionActive && (
        <button
          onClick={handleTranceToggle}
          style={{
            marginTop      : 8,
            width          : "100%",
            padding        : "7px 0",
            borderRadius   : 5,
            border         : "none",
            cursor         : "pointer",
            backgroundColor: tranceMode
              ? "rgba(0,230,118,0.15)"
              : "rgba(255,255,255,0.05)",
            color          : tranceMode ? "#00e676" : "rgba(255,255,255,0.4)",
            fontFamily     : "'JetBrains Mono', monospace",
            fontSize       : 10,
            letterSpacing  : "0.1em",
            outline        : `1px solid ${tranceMode ? "rgba(0,230,118,0.3)" : "rgba(255,255,255,0.1)"}`,
            transition     : "all 0.2s",
          } as React.CSSProperties}
        >
          {tranceMode ? "◈  MODO TRANCE  ON" : "◇  MODO TRANCE  OFF"}
        </button>
      )}

      {/* NRS-T (Numeric Rating Scale — Trance) */}
      {isSessionActive && (
        <button
          onClick={() => onSubmitNRS(sessionId ?? "", 7)}
          style={{
            marginTop      : 8,
            width          : "100%",
            padding        : "7px 0",
            borderRadius   : 5,
            border         : "none",
            cursor         : "pointer",
            backgroundColor: "rgba(255,214,0,0.1)",
            color          : "#ffd600",
            fontFamily     : "'JetBrains Mono', monospace",
            fontSize       : 10,
            letterSpacing  : "0.1em",
            outline        : "1px solid rgba(255,214,0,0.25)",
            transition     : "all 0.2s",
          } as React.CSSProperties}
          title="Enviar medida subjetiva NRS-T (demo valor=7)"
        >
          ◎  NRS-T (demo)
        </button>
      )}

      {/* Ping — diagnóstico de latencia WS */}
      <button
        onClick={onPing}
        disabled={!isConnected}
        style={{
          marginTop      : 8,
          width          : "100%",
          padding        : "5px 0",
          borderRadius   : 5,
          border         : "none",
          cursor         : isConnected ? "pointer" : "not-allowed",
          backgroundColor: "rgba(255,255,255,0.04)",
          color          : isConnected ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.15)",
          fontFamily     : "'JetBrains Mono', monospace",
          fontSize       : 9,
          letterSpacing  : "0.1em",
          outline        : "1px solid rgba(255,255,255,0.08)",
          transition     : "all 0.2s",
        } as React.CSSProperties}
      >
        ◌  PING WS
      </button>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Info de sesión activa */}
      {isSessionActive && (
        <div style={{
          padding      : "10px 12px",
          borderRadius : 4,
          background   : "rgba(0,229,255,0.05)",
          border       : "1px solid rgba(0,229,255,0.12)",
          marginTop    : 12,
        }}>
          <div style={{
            fontSize : 8,
            fontFamily: "'JetBrains Mono', monospace",
            color    : "rgba(0,229,255,0.6)",
            marginBottom: 6,
            letterSpacing: "0.1em",
          }}>
            SESIÓN ACTIVA
          </div>
          <div style={{
            fontSize : 10,
            fontFamily: "'JetBrains Mono', monospace",
            color    : "rgba(255,255,255,0.6)",
          }}>
            {patientId || "—"} · {group === "experimental" ? "EXP" : "CTR"}
          </div>
        </div>
      )}

      {/* z-score rápido */}
      {/* REMOVIDO: ZScoreWidget (reemplazado por State Classification) */}
    </div>
  );
}

function ValidationPanel() {
  const frontalSpecificity = useEEGStore(selectFrontalSpecificity);
  const frontalValid = useEEGStore(selectFrontalSpecificityValid);
  const artifact = useEEGStore(selectArtifactDetected);

  const cellStyle: React.CSSProperties = {
    padding: "8px 10px",
    borderRadius: 6,
    border: "1px solid rgba(255,255,255,0.12)",
    fontFamily: "'JetBrains Mono', monospace",
    fontSize: 10,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 10,
  };

  return (
    <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
      <div style={{
        fontSize: 9,
        fontFamily: "'JetBrains Mono', monospace",
        letterSpacing: "0.12em",
        color: "rgba(255,255,255,0.3)",
      }}>
        VALIDACION DE SEGMENTO
      </div>

      <div style={{
        ...cellStyle,
        background: frontalValid ? "rgba(0,230,118,0.1)" : "rgba(255,82,82,0.1)",
        borderColor: frontalValid ? "rgba(0,230,118,0.25)" : "rgba(255,82,82,0.25)",
      }}>
        <span>Especificidad frontal</span>
        <span style={{ color: frontalValid ? "#00e676" : "#ff5252" }}>
          {frontalValid ? "VALIDA" : "INVALIDA"}
        </span>
      </div>

      <div style={{
        ...cellStyle,
        background: artifact ? "rgba(255,82,82,0.1)" : "rgba(0,229,255,0.1)",
        borderColor: artifact ? "rgba(255,82,82,0.25)" : "rgba(0,229,255,0.25)",
      }}>
        <span>Artefacto ocular</span>
        <span style={{ color: artifact ? "#ff5252" : "#00e5ff" }}>
          {artifact ? "CONTAMINADO" : "LIMPIO"}
        </span>
      </div>

      <div style={{
        ...cellStyle,
        background: "rgba(255,255,255,0.03)",
        borderColor: "rgba(255,255,255,0.1)",
      }}>
        <span>Fz / media(F3,F4)</span>
        <span style={{ color: "#ffd600" }}>{frontalSpecificity.toFixed(2)}</span>
      </div>
    </div>
  );
}

// ========== Panels para los nuevos componentes del clasificador ==========

function StateClassifierPanel() {
  const prediction = useEEGStore(selectStatePrediction);
  
  return (
    <StateClassifier
      label={prediction?.predicted_label ?? null}
      confidence={prediction?.confidence ?? 0}
      timestamp={prediction ? Date.now() : undefined}
    />
  );
}

function ConfidenceGaugePanel() {
  const prediction = useEEGStore(selectStatePrediction);
  
  return (
    <ConfidenceGauge
      confidence={prediction?.confidence ?? 0}
    />
  );
}

function ProbabilityMatrixPanel() {
  const prediction = useEEGStore(selectStatePrediction);
  
  return (
    <ProbabilityMatrix
      awake={prediction?.class_probabilities.awake ?? 0}
      induction={prediction?.class_probabilities.induction ?? 0}
      trance={prediction?.class_probabilities.trance ?? 0}
    />
  );
}

// ---------------------------------------------------------------------------
// App principal
// ---------------------------------------------------------------------------

export default function App() {
  const {
    isConnected,
    isSessionActive,
    reconnectAttempt,
    sendMessage,
    startSession,
    stopSession,
    setTranceMode,
    submitSubjective,
    setInspectionChannel,
    loadDataset,
    dataset,
  } = useEEGSocket();

  const lastError = useEEGStore(selectLastError);
  const sessionId = useEEGStore((s) => s.sessionId);
  const inspectionChannel = useEEGStore((s) => s.inspectionChannel);

  return (
    <div style={{
      minHeight      : "100vh",
      backgroundColor: "#080810",
      color          : "rgba(255,255,255,0.85)",
      fontFamily     : "'JetBrains Mono', monospace",
      display        : "flex",
      flexDirection  : "column",
      padding        : "12px 16px",
      gap            : 12,
      boxSizing      : "border-box",
    }}>

      {/* ── ERROR BANNER ────────────────────────────────────────────────── */}
      {lastError && (
        <div style={{
          padding        : "10px 16px",
          backgroundColor: "rgba(255,23,68,0.12)",
          border         : "1px solid rgba(255,23,68,0.35)",
          borderRadius   : 6,
          fontSize       : 11,
          fontFamily     : "'JetBrains Mono', monospace",
          color          : "#ff5252",
          display        : "flex",
          alignItems     : "center",
          gap            : 8,
        }}>
          <span>⚠</span>
          <span>{lastError}</span>
        </div>
      )}

      {/* ── HEADER ──────────────────────────────────────────────────────── */}
      <header style={{
        display        : "flex",
        alignItems     : "center",
        justifyContent : "space-between",
        padding        : "10px 16px",
        backgroundColor: "#0d0d1a",
        borderRadius   : 8,
        border         : "1px solid rgba(255,255,255,0.07)",
      }}>
        {/* Logo / título */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width          : 32,
            height         : 32,
            borderRadius   : "50%",
            border         : "2px solid #00e5ff",
            display        : "flex",
            alignItems     : "center",
            justifyContent : "center",
            fontSize       : 14,
            color          : "#00e5ff",
          }}>
            ψ
          </div>
          <div>
            <div style={{
              fontSize     : 13,
              fontWeight   : 700,
              letterSpacing: "0.08em",
              color        : "rgba(255,255,255,0.9)",
            }}>
              NEUROFEEDBACK EEG
            </div>
            <div style={{
              fontSize     : 8,
              color        : "rgba(255,255,255,0.3)",
              letterSpacing: "0.12em",
              marginTop    : 1,
            }}>
              SISTEMA DE MONITOREO EN TIEMPO REAL  ·  250 SPS
            </div>
          </div>
        </div>

        {/* Centro: predicción de estado */}
        <StatePredictionBadge />

        {/* Derecha: conexión + latencia */}
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <PipelineLatency />
          <ConnectionDot connected={isConnected} retryAttempt={reconnectAttempt} />
        </div>
      </header>

      {/* ── WAVEFORM (ancho completo) ────────────────────────────────────── */}
      <section style={{
        backgroundColor: "#0d0d1a",
        borderRadius   : 8,
        border         : "1px solid rgba(255,255,255,0.07)",
        padding        : "12px 14px",
      }}>
        <WaveformChart sampleRate={SAMPLE_RATE} height={200} />
      </section>

      <ChannelInspectionPanel />

      {/* ── FILA INFERIOR: Clasificador + Visualizaciones ──────────────────── */}
      <div style={{
        display             : "grid",
        gridTemplateColumns : "180px 180px 200px 1fr 260px",
        gap                 : 12,
        flex                : 1,
        minHeight           : 0,
      }}>

        {/* Panel 1 — State Classifier */}
        <section style={{
          backgroundColor: "#0d0d1a",
          borderRadius   : 8,
          border         : "1px solid rgba(255,255,255,0.07)",
          padding        : "12px",
          display        : "flex",
          alignItems     : "center",
          justifyContent : "center",
        }}>
          <StateClassifierPanel />
        </section>

        {/* Panel 2 — Confidence Gauge */}
        <section style={{
          backgroundColor: "#0d0d1a",
          borderRadius   : 8,
          border         : "1px solid rgba(255,255,255,0.07)",
          padding        : "12px",
          display        : "flex",
          alignItems     : "center",
          justifyContent : "center",
        }}>
          <ConfidenceGaugePanel />
        </section>

        {/* Panel 3 — Probability Matrix */}
        <section style={{
          backgroundColor: "#0d0d1a",
          borderRadius   : 8,
          border         : "1px solid rgba(255,255,255,0.07)",
          padding        : "12px",
          display        : "flex",
          alignItems     : "center",
          justifyContent : "center",
        }}>
          <ProbabilityMatrixPanel />
        </section>

        {/* Panel 4 — Band Power + ThetaBeta */}
        <div style={{
          display: "grid",
          gridTemplateRows: "1fr 1fr",
          gap: 12,
        }}>
          <section style={{
            backgroundColor: "#0d0d1a",
            borderRadius   : 8,
            border         : "1px solid rgba(255,255,255,0.07)",
            padding        : "16px 18px",
            display        : "flex",
            flexDirection  : "column",
            justifyContent : "center",
          }}>
            <BandPowerBars />
          </section>

          <section style={{
            backgroundColor: "#0d0d1a",
            borderRadius   : 8,
            border         : "1px solid rgba(255,255,255,0.07)",
            padding        : "16px 12px",
            display        : "flex",
            alignItems     : "center",
            justifyContent : "center",
          }}>
            <ThetaBetaGauge />
          </section>
        </div>

        {/* Panel 5 — Control + Topography */}
        <section style={{ minHeight: 0 }}>
          <ControlPanel
            isConnected={isConnected}
            isSessionActive={isSessionActive}
            sessionId={sessionId}
            onStart={startSession}
            onStop={stopSession}
            onTranceToggle={setTranceMode}
            onSubmitNRS={(sid, value) =>
              submitSubjective({
                sessionId : sid,
                timestamp : Date.now(),
                label     : "nrs-t-demo",
                value,
                notes     : "Demo subjective measure",
              })
            }
            onPing={() => sendMessage({ type: "ping" })}
            inspectionChannel={inspectionChannel}
            onInspectionChannelChange={setInspectionChannel}
            onLoadDataset={loadDataset}
            dataset={dataset}
          />
        </section>
      </div>

      {/* ── FOOTER ──────────────────────────────────────────────────────── */}
      <footer style={{
        textAlign    : "center",
        fontSize     : 8,
        color        : "rgba(255,255,255,0.15)",
        letterSpacing: "0.1em",
        paddingTop   : 4,
      }}>
        Filtros: Notch 50 Hz · Butterworth BP 1–30 Hz · FFT 256pt · Welford Z-Score · Sigmoid feedback
      </footer>
    </div>
  );
}
