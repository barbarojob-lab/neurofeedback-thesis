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
  useEffect,
  useMemo,
} from "react";

import ThetaBetaGauge   from "./components/ThetaBetaGauge";
import BandPowerBars    from "./components/BandPowerBars";
// ✅ NUEVOS: Componentes del clasificador
import StateClassifier  from "./components/StateClassifier";
import { useEEGSocket } from "./hooks/useEEGSocket";
import {
  useEEGStore,
  selectPipelineMs,
  selectLastError,
  selectStatePrediction,
} from "./store/eegStore";

import type { SessionConfig, DatasetMetadata, PlaybackInfo } from "./types";

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

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
    uncertain: "#90a4ae",
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
  onStart           : (cfg: SessionConfig) => void;
  onStop            : () => void;
  onPing            : () => void;
  inspectionChannel : string;
  onInspectionChannelChange: (channel: string) => void;
  onLoadDataset: (path: string) => void;
  onSetPlaybackPosition: (seconds: number) => void;
  dataset: DatasetMetadata | null;
  playback: PlaybackInfo | null;
}

function ControlPanel({
  isConnected,
  isSessionActive,
  onStart,
  onStop,
  onPing,
  inspectionChannel,
  onInspectionChannelChange,
  onLoadDataset,
  onSetPlaybackPosition,
  dataset,
  playback,
}: ControlPanelProps) {
  const [patientId,    setPatientId]    = useState("");
  const [group,        setGroup]        = useState<"experimental" | "control">("experimental");
  const [sigmoidK,     setSigmoidK]     = useState(2.0);
  const [thetaThresh,  setThetaThresh]  = useState(0.0);
  const [datasetPath, setDatasetPath] = useState("");
  const [datasetOptions, setDatasetOptions] = useState<string[]>([]);
  const [datasetQuery, setDatasetQuery] = useState("");
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [seekSecondsInput, setSeekSecondsInput] = useState("0");

  const patientIdId = useId();
  const groupId     = useId();

  const inferModelProfileMode = useCallback((): "auto" | "high" | "low" => {
    const source = (dataset?.sourcePath ?? datasetPath).toLowerCase();
    if (source.includes("_high") || source.includes("high")) return "high";
    if (source.includes("_low") || source.includes("low")) return "low";
    return "auto";
  }, [dataset?.sourcePath, datasetPath]);

  const handleStart = useCallback(() => {
    const modelProfileMode = inferModelProfileMode();
    const config: SessionConfig = {
      thetaThreshold        : thetaThresh,
      thetaPeakTbrThreshold : 2.5,
      sigmoidK,
      modelProfileMode,
    };
    onStart(config);
  }, [onStart, thetaThresh, sigmoidK, inferModelProfileMode]);

  const refreshDatasets = useCallback(async () => {
    setLoadingDatasets(true);
    setDatasetsError(null);
    try {
      const res = await fetch("http://localhost:8080/datasets");
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const body = (await res.json()) as { datasets?: string[] };
      const list = Array.isArray(body.datasets) ? body.datasets : [];
      setDatasetOptions(list);

      const firstMatch = list.find((p) => p.toLowerCase().includes(datasetQuery.toLowerCase()));
      if (!datasetPath && firstMatch) {
        setDatasetPath(firstMatch);
      }
    } catch (err) {
      setDatasetsError((err as Error).message);
      setDatasetOptions([]);
    } finally {
      setLoadingDatasets(false);
    }
  }, [datasetPath, datasetQuery]);

  useEffect(() => {
    if (isConnected) {
      void refreshDatasets();
    }
  }, [isConnected, refreshDatasets]);

  const filteredDatasetOptions = datasetQuery.trim().length === 0
    ? datasetOptions
    : datasetOptions.filter((p) => p.toLowerCase().includes(datasetQuery.toLowerCase()));

  useEffect(() => {
    if (filteredDatasetOptions.length === 0) {
      return;
    }
    if (!filteredDatasetOptions.includes(datasetPath)) {
      setDatasetPath(filteredDatasetOptions[0] ?? "");
    }
  }, [filteredDatasetOptions, datasetPath]);

  useEffect(() => {
    if (!playback) return;
    setSeekSecondsInput(String(Math.round(playback.positionSec)));
  }, [playback?.durationSec]);

  const clampedPlaybackSeconds = useMemo(() => {
    if (!playback) return 0;
    return Math.max(0, Math.min(playback.durationSec, playback.positionSec));
  }, [playback]);

  const seekToInput = useCallback(() => {
    if (!playback) return;
    const parsed = Number(seekSecondsInput);
    if (!Number.isFinite(parsed)) return;
    const bounded = Math.max(0, Math.min(playback.durationSec, parsed));
    onSetPlaybackPosition(bounded);
  }, [onSetPlaybackPosition, playback, seekSecondsInput]);

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

      {field("DATASET (busqueda)",
        <div style={{ display: "grid", gap: 6 }}>
          <div style={{ display: "flex", gap: 6 }}>
            <input
              type="text"
              value={datasetQuery}
              onChange={(e) => setDatasetQuery(e.target.value)}
              placeholder="Buscar por nombre: subject, sesion, etc."
              style={{ ...inputStyle, flex: 1 }}
            />
            <button
              onClick={() => void refreshDatasets()}
              disabled={!isConnected || loadingDatasets}
              style={{
                padding: "6px 10px",
                borderRadius: 5,
                border: "none",
                cursor: !isConnected || loadingDatasets ? "not-allowed" : "pointer",
                backgroundColor: "rgba(255,255,255,0.08)",
                color: "rgba(255,255,255,0.85)",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
              } as React.CSSProperties}
            >
              {loadingDatasets ? "BUSCANDO..." : "REFRESH"}
            </button>
          </div>

          <select
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            disabled={!isConnected || filteredDatasetOptions.length === 0}
            style={inputStyle}
          >
            {filteredDatasetOptions.length === 0 && (
              <option value="">
                {loadingDatasets ? "Detectando datasets..." : "Sin resultados (EDF/CSV)"}
              </option>
            )}
            {filteredDatasetOptions.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>

          <input
            type="text"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            placeholder="Ruta manual opcional"
            style={inputStyle}
          />

          {datasetsError && (
            <div style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 10,
              color: "#ff8a80",
            }}>
              Error listando datasets: {datasetsError}
            </div>
          )}

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

      {playback && (
        <div style={{
          padding: "10px",
          borderRadius: 6,
          marginBottom: 10,
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.12)",
          display: "grid",
          gap: 8,
        }}>
          <div style={{
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 9,
            color: "rgba(255,255,255,0.6)",
            letterSpacing: "0.08em",
          }}>
            NAVEGACION TEMPORAL
          </div>

          <input
            type="range"
            min={0}
            max={Math.max(1, playback.durationSec)}
            step={1}
            value={clampedPlaybackSeconds}
            onChange={(e) => {
              const next = Number(e.target.value);
              setSeekSecondsInput(String(Math.round(next)));
              onSetPlaybackPosition(next);
            }}
            style={{ width: "100%", accentColor: "#00e5ff" }}
          />

          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr auto auto",
            gap: 6,
            alignItems: "center",
          }}>
            <input
              type="number"
              min={0}
              max={Math.max(1, playback.durationSec)}
              step={1}
              value={seekSecondsInput}
              onChange={(e) => setSeekSecondsInput(e.target.value)}
              style={inputStyle}
            />
            <button
              onClick={seekToInput}
              style={{
                padding: "7px 10px",
                borderRadius: 5,
                border: "none",
                cursor: "pointer",
                backgroundColor: "rgba(0,229,255,0.16)",
                color: "#00e5ff",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                letterSpacing: "0.08em",
              }}
            >
              IR
            </button>
            <button
              onClick={() => onSetPlaybackPosition(0)}
              style={{
                padding: "7px 10px",
                borderRadius: 5,
                border: "none",
                cursor: "pointer",
                backgroundColor: "rgba(255,255,255,0.1)",
                color: "rgba(255,255,255,0.8)",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 10,
                letterSpacing: "0.08em",
              }}
            >
              INICIO
            </button>
          </div>
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

// ========== Panels para los componentes principales ==========

function StateClassifierPanel() {
  const prediction = useEEGStore(selectStatePrediction);
  
  return (
    <StateClassifier
      label={prediction?.predicted_label ?? null}
      confidence={prediction?.confidence ?? 0}
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
    setInspectionChannel,
    loadDataset,
    dataset,
    playback,
    setPlaybackPosition,
  } = useEEGSocket();

  const lastError = useEEGStore(selectLastError);
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

      {/* ── Panel superior (navegacion temporal) ───────────────────────── */}
      <section style={{
        backgroundColor: "#0d0d1a",
        borderRadius   : 8,
        border         : "1px solid rgba(255,255,255,0.07)",
        padding        : "12px 14px",
      }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: "auto 1fr auto",
          alignItems: "center",
          gap: 12,
        }}>
          <div style={{
            fontSize: 10,
            color: "rgba(255,255,255,0.45)",
            letterSpacing: "0.1em",
          }}>
            REPRODUCCION DATASET
          </div>
          <div style={{
            height: 10,
            borderRadius: 999,
            background: "rgba(255,255,255,0.09)",
            overflow: "hidden",
          }}>
            <div style={{
              width: playback && playback.durationSec > 0
                ? `${Math.min(100, Math.max(0, (playback.positionSec / playback.durationSec) * 100))}%`
                : "0%",
              height: "100%",
              background: "linear-gradient(90deg, #00e5ff, #2ef5c8)",
              transition: "width 0.15s linear",
            }} />
          </div>
          <div style={{
            fontSize: 10,
            color: "rgba(255,255,255,0.8)",
            minWidth: 120,
            textAlign: "right",
          }}>
            {playback
              ? `${playback.positionSec.toFixed(1)}s / ${playback.durationSec.toFixed(1)}s`
              : "Sin dataset"}
          </div>
        </div>
      </section>

      {/* ── FILA INFERIOR: Clasificador + Visualizaciones ──────────────────── */}
      <div style={{
        display             : "grid",
        gridTemplateColumns : "180px 1fr 260px",
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

        {/* Panel 2 — Band Power + ThetaBeta */}
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

        {/* Panel 3 — Control */}
        <section style={{ minHeight: 0 }}>
          <ControlPanel
            isConnected={isConnected}
            isSessionActive={isSessionActive}
            onStart={startSession}
            onStop={stopSession}
            onPing={() => sendMessage({ type: "ping" })}
            inspectionChannel={inspectionChannel}
            onInspectionChannelChange={setInspectionChannel}
            onLoadDataset={loadDataset}
            onSetPlaybackPosition={setPlaybackPosition}
            dataset={dataset}
            playback={playback}
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
        Filtros: Notch 50 Hz · Butterworth BP 1–30 Hz · FFT 256pt · Welford Z-Score
      </footer>
    </div>
  );
}
