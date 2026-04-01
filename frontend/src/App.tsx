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
import TranceDepth      from "./components/TranceDepth";
import BandPowerBars    from "./components/BandPowerBars";
import { useEEGSocket } from "./hooks/useEEGSocket";
import {
  useEEGStore,
  selectCommand,
  selectPipelineMs,
  selectZScore,
  selectLastError,
} from "./store/eegStore";

import type { SessionConfig } from "./types";

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

const SAMPLE_RATE = 250;

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

/** Badge de comando de feedback actual */
function CommandBadge() {
  const command = useEEGStore(selectCommand);
  if (!command) return null;

  const styles: Record<string, { bg: string; color: string; label: string }> = {
    decrease_theta : { bg: "rgba(255,23,68,0.15)",  color: "#ff5252", label: "↓ REDUCIR THETA" },
    neutral        : { bg: "rgba(96,125,139,0.15)", color: "#90a4ae", label: "◆ NEUTRO"         },
    increase_theta : { bg: "rgba(0,230,118,0.15)",  color: "#00e676", label: "↑ AUMENTAR THETA" },
    sustain_trance : { bg: "rgba(0,229,255,0.15)",  color: "#00e5ff", label: "◈ MANTENER TRANCE"},
  };
  const s = styles[command.action] ?? styles.neutral;

  return (
    <div style={{
      display      : "flex",
      alignItems   : "center",
      gap          : 8,
      padding      : "4px 12px",
      borderRadius : 4,
      background   : s.bg,
      border       : `1px solid ${s.color}33`,
    }}>
      <span style={{
        fontSize     : 10,
        fontFamily   : "'JetBrains Mono', monospace",
        color        : s.color,
        letterSpacing: "0.1em",
        fontWeight   : 600,
      }}>
        {s.label}
      </span>
      <span style={{
        fontSize : 10,
        color    : "rgba(255,255,255,0.3)",
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        {(command.intensity * 100).toFixed(0)}%
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
}: ControlPanelProps) {
  const [patientId,    setPatientId]    = useState("");
  const [group,        setGroup]        = useState<"experimental" | "control">("experimental");
  const [tranceMode,   setTranceModeLocal] = useState(false);
  const [sigmoidK,     setSigmoidK]     = useState(2.0);
  const [thetaThresh,  setThetaThresh]  = useState(0.0);

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
          border_color   : isSessionActive ? "#ff524244" : "#00e5ff44",
          fontFamily     : "'JetBrains Mono', monospace",
          fontSize       : 12,
          fontWeight     : 700,
          letterSpacing  : "0.12em",
          transition     : "all 0.2s",
          marginTop      : 4,
          outline        : "1px solid",
          outlineColor   : isSessionActive
            ? "rgba(255,82,82,0.35)"
            : isConnected ? "rgba(0,229,255,0.3)" : "rgba(255,255,255,0.1)",
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
      <ZScoreWidget />
    </div>
  );
}

/** Widget compacto del z-score actual para el panel de control */
function ZScoreWidget() {
  const z = useEEGStore(selectZScore);
  const color = z > 1 ? "#00e676" : z > 0 ? "#00e5ff" : z > -1 ? "#90a4ae" : "#ff5252";

  return (
    <div style={{
      marginTop   : 10,
      textAlign   : "center",
      padding     : "8px 0",
      borderTop   : "1px solid rgba(255,255,255,0.06)",
    }}>
      <div style={{
        fontSize : 8,
        fontFamily: "'JetBrains Mono', monospace",
        color    : "rgba(255,255,255,0.25)",
        letterSpacing: "0.1em",
        marginBottom: 4,
      }}>
        Z-SCORE ACTUAL
      </div>
      <div style={{
        fontSize   : 22,
        fontFamily : "'JetBrains Mono', monospace",
        fontWeight : 700,
        color,
        transition : "color 0.3s",
      }}>
        {z >= 0 ? "+" : ""}{z.toFixed(2)}σ
      </div>
    </div>
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
  } = useEEGSocket();

  const lastError = useEEGStore(selectLastError);
  const { sessionId } = useEEGStore(s => ({ sessionId: s.sessionId }));

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

        {/* Centro: comando activo */}
        <CommandBadge />

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

      {/* ── FILA INFERIOR: 4 paneles ─────────────────────────────────────── */}
      <div style={{
        display             : "grid",
        gridTemplateColumns : "220px 130px 1fr 220px",
        gap                 : 12,
        flex                : 1,
        minHeight           : 0,
      }}>

        {/* Panel 1 — ThetaBeta Gauge */}
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

        {/* Panel 2 — Trance Depth */}
        <section style={{
          backgroundColor: "#0d0d1a",
          borderRadius   : 8,
          border         : "1px solid rgba(255,255,255,0.07)",
          padding        : "14px 8px",
          display        : "flex",
          alignItems     : "center",
          justifyContent : "center",
        }}>
          <TranceDepth />
        </section>

        {/* Panel 3 — Band Power Bars */}
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

        {/* Panel 4 — Control */}
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
