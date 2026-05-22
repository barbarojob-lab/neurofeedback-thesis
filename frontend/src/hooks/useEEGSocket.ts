/**
 * useEEGSocket.ts  —  frontend/src/hooks/
 *
 * Hook React que gestiona la conexión WebSocket al servidor de neurofeedback.
 *
 * ── Responsabilidades ────────────────────────────────────────────────────
 *
 *   1. Conectar a ws://localhost:8080 al montar el componente.
 *   2. Parsear mensajes JSON del servidor y despacharlos al store Zustand.
 *   3. Reconectar automáticamente con backoff exponencial ante desconexiones.
 *   4. Cerrar la conexión limpiamente en unmount y en beforeunload.
 *   5. Exponer API de alto nivel: startSession, stopSession, setTranceMode.
 *
 * ── Backoff exponencial ──────────────────────────────────────────────────
 *
 *   Intento 0: espera 1 s
 *   Intento 1: espera 2 s
 *   Intento 2: espera 4 s
 *   Intento 3: espera 8 s
 *   Intento 4: espera 16 s
 *   Intento 5+: espera 30 s (techo)
 *
 *   Fórmula: delay = min(BASE_DELAY_MS × 2^attempt, MAX_DELAY_MS)
 *
 *   Jitter opcional (no implementado): añadir ±20 % aleatorio al delay para
 *   evitar thundering herd si muchos clientes reconectan simultáneamente.
 *   Relevante solo si hay múltiples instancias del frontend (poco probable
 *   en el contexto clínico de un solo terapeuta).
 *
 * ── Separación pushWaveformSample / updateMetrics ────────────────────────
 *
 *   Cada mensaje `eeg_data` contiene AMBOS: el filteredSample más reciente
 *   Y las métricas del epoch. El hook despacha:
 *     1. pushWaveformSample(payload.filteredSample) — O(1), sin re-render
 *     2. updateMetrics(payload) — un setState agrupa todas las métricas
 *
 *   Para mantener la ilusión de 250 samples/s en el canvas mientras solo
 *   recibimos 4 payloads/s, el servidor podría enviar un buffer de 64
 *   muestras por mensaje. Si se implementa esa optimización, el hook
 *   debería llamar pushWaveformSample 64 veces por mensaje en un loop.
 *   Por ahora, un sample por mensaje es suficiente para la demo.
 *
 * ── Por qué useRef para el WebSocket (no useState) ───────────────────────
 *
 *   - `wsRef.current` es la conexión activa. No necesitamos re-renderizar
 *     el componente cuando la referencia cambia (reconexión interna).
 *   - `reconnectAttemptRef` y `reconnectTimerRef` tampoco deben causar
 *     renders — son estado interno del mecanismo de reconexión.
 *   - Solo `isConnected` es estado de React (via el store Zustand) porque
 *     la UI necesita mostrar el indicador de conexión.
 */

import {
  useEffect,
  useRef,
  useCallback,
  useState,
} from "react";

import { useEEGStore } from "../store/eegStore";

import type {
  WSMessage,
  ServerMessage,
  SessionConfig,
  SubjectiveMeasure,
  DatasetMetadata,
  PlaybackInfo,
} from "../types";

import {
  isFeedbackPayload,
  isServerHello,
  isSessionStarted,
  isSessionStopped,
  isServerError,
  isInspectionChannelSet,
  isDatasetLoaded,
  isPlaybackPositionSet,
} from "../types";

// ---------------------------------------------------------------------------
// Constantes del backoff
// ---------------------------------------------------------------------------

/**
 * Calcula la URL del WebSocket con fallback inteligente:
 * 1. Si hay VITE_WS_URL en .env, usar ese valor directamente
 * 2. Si la página está en HTTPS, usar WSS (WebSocket Secure)
 * 3. Si está en HTTP, usar WS
 * 4. En localhost siempre usar WS (desarrollo local sin certificado)
 *
 * Ejemplos:
 *   localhost (dev):           ws://localhost:8080
 *   https://example.com/app:   wss://example.com:8080
 *   http://example.com/app:    ws://example.com:8080
 */
function resolveWsUrl(): string {
  const envUrl = import.meta.env.VITE_WS_URL;
  if (envUrl) return envUrl;

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.hostname;
  const port = 8080; // puerto default del servidor neurofeedback

  // En localhost, no usar el puerto configurado — el dev server redirige
  if (host === "localhost" || host === "127.0.0.1") {
    return `ws://localhost:${port}`;
  }

  return `${protocol}://${host}:${port}`;
}

const WS_URL        = resolveWsUrl();
const BASE_DELAY_MS = 1_000;   // 1 s — primer intento de reconexión
const MAX_DELAY_MS  = 30_000;  // 30 s — techo del backoff
const MAX_ATTEMPTS  = 10;      // abandonar tras 10 intentos fallidos consecutivos

// ---------------------------------------------------------------------------
// Tipo del hook
// ---------------------------------------------------------------------------

export interface UseEEGSocketReturn {
  /** true cuando el WebSocket está en estado OPEN */
  isConnected     : boolean;
  /** true cuando hay una sesión EEG activa */
  isSessionActive : boolean;
  /** Intento de reconexión actual (0 = conectado o primer intento) */
  reconnectAttempt: number;
  /** Envía cualquier WSMessage al servidor */
  sendMessage     : (msg: WSMessage) => void;
  /** Inicia una sesión EEG con la configuración dada */
  startSession    : (config?: SessionConfig) => void;
  /** Detiene la sesión EEG activa */
  stopSession     : () => void;
  /** Activa o desactiva el modo trance en el simulador del servidor */
  setTranceMode   : (enabled: boolean) => void;
  /** Envía una medida subjetiva a la base de datos del servidor */
  submitSubjective: (measure: SubjectiveMeasure) => void;
  /** Selecciona el canal para el panel de inspección */
  setInspectionChannel: (channel: string) => void;
  /** Carga metadata de dataset EEG desde ruta local del backend */
  loadDataset: (filePath: string) => void;
  /** Último dataset cargado */
  dataset: DatasetMetadata | null;
  /** Posición actual de reproducción cuando la fuente es dataset */
  playback: PlaybackInfo | null;
  /** Mueve el cursor de reproducción al segundo indicado */
  setPlaybackPosition: (seconds: number) => void;
}

// ---------------------------------------------------------------------------
// Hook principal
// ---------------------------------------------------------------------------

export function useEEGSocket(): UseEEGSocketReturn {
  // ── Refs internos (sin causar re-render) ─────────────────────────────
  const wsRef                = useRef<WebSocket | null>(null);
  const reconnectAttemptRef  = useRef<number>(0);
  const reconnectTimerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMountedRef         = useRef<boolean>(true);
  const reconnectAttemptState= useRef<number>(0); // para exponer al consumidor
  const [dataset, setDataset] = useState<DatasetMetadata | null>(null);
  const [playback, setPlayback] = useState<PlaybackInfo | null>(null);

  // ── Acciones/estado del store con selectores atómicos ────────────────
  const pushWaveformSample = useEEGStore((s) => s.pushWaveformSample);
  const updateMetrics = useEEGStore((s) => s.updateMetrics);
  const setConnected = useEEGStore((s) => s.setConnected);
  const setSessionActive = useEEGStore((s) => s.setSessionActive);
  const setSessionConfig = useEEGStore((s) => s.setSessionConfig);
  const setError = useEEGStore((s) => s.setError);
  const setInspectionChannelState = useEEGStore((s) => s.setInspectionChannel);
  const resetSession = useEEGStore((s) => s.resetSession);
  const isConnected = useEEGStore((s) => s.isConnected);
  const isSessionActive = useEEGStore((s) => s.isSessionActive);

  // ── Calcular delay de backoff ─────────────────────────────────────────
  const getBackoffDelay = useCallback((attempt: number): number => {
    return Math.min(BASE_DELAY_MS * Math.pow(2, attempt), MAX_DELAY_MS);
  }, []);

  // ── Manejador de mensajes entrantes ──────────────────────────────────
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      let msg: ServerMessage;

      try {
        msg = JSON.parse(event.data as string) as ServerMessage;
      } catch {
        console.error("[useEEGSocket] JSON inválido:", event.data);
        return;
      }

      // ── Despacho por tipo usando type guards ──────────────────────────
      // Los type guards validan el campo `type` en runtime, evitando
      // casteos ciegos que podrían crashear si el servidor cambia su API.

      if (isFeedbackPayload(msg)) {
        // Hot path — ~4 veces/segundo
        // 1. Todos los samples del hop → ring buffer (O(n), sin re-render)
        //    filteredSamples tiene hopSize=64 muestras → 250 sps reales en el osciloscopio
        const samples = msg.filteredSamples ?? [msg.filteredSample];
        for (let i = 0; i < samples.length; i++) {
          pushWaveformSample(samples[i]!);
        }
        // 2. Métricas del epoch (un setState agrupa todo)
        updateMetrics(msg);
        setPlayback(msg.playback ?? null);
        return;
      }

      if (isServerHello(msg)) {
        console.log(
          `[useEEGSocket] Server hello — ` +
          `v${msg.version}, ${msg.sampleRate} sps, ` +
          `win=${msg.windowSize}, hop=${msg.hopSize}`
        );
        // Restaurar estado de sesión si el servidor tenía una sesión activa
        if (msg.session) {
          setSessionActive(true, msg.session.id);
        }
        setDataset(msg.dataset ?? null);
        setPlayback(msg.playback ?? null);
        return;
      }

      if (isSessionStarted(msg)) {
        setSessionActive(true, msg.sessionId);
        console.log(`[useEEGSocket] Sesión iniciada: ${msg.sessionId}`);
        return;
      }

      if (isSessionStopped(msg)) {
        setSessionActive(false);
        resetSession();
        setPlayback(null);
        console.log("[useEEGSocket] Sesión detenida.");
        return;
      }

      if (isServerError(msg)) {
        setError(msg.message);
        return;
      }

      if (isInspectionChannelSet(msg)) {
        setInspectionChannelState(msg.channel);
        return;
      }

      if (isDatasetLoaded(msg)) {
        setDataset(msg.dataset);
        setPlayback(msg.playback ?? null);
        return;
      }

      if (isPlaybackPositionSet(msg)) {
        setPlayback({
          positionSec: msg.positionSec,
          durationSec: msg.durationSec,
        });
        return;
      }

      // Mensajes informativos que no requieren acción en el store
      switch (msg.type) {
        case "pong":
          // Latencia round-trip: Date.now() - msg.timestamp
          break;
        case "subjective_saved":
          console.log(`[useEEGSocket] Medida subjetiva guardada: ${msg.sessionId}`);
          break;
        case "trance_mode_set":
          console.log(`[useEEGSocket] Trance mode: ${msg.enabled ? "ON" : "OFF"}`);
          break;
        default:
          console.warn("[useEEGSocket] Mensaje no reconocido:", (msg as { type: string }).type);
      }
    },
    [pushWaveformSample, updateMetrics, setSessionActive, setError, setInspectionChannelState, resetSession]
  );

  // ── Conectar WebSocket ────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (!isMountedRef.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (wsRef.current?.readyState === WebSocket.CONNECTING) return;

    console.log(
      `[useEEGSocket] Conectando a ${WS_URL}` +
      (reconnectAttemptRef.current > 0
        ? ` (intento ${reconnectAttemptRef.current})`
        : "")
    );

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!isMountedRef.current) { ws.close(); return; }

      console.log("[useEEGSocket] ✅ Conectado");
      reconnectAttemptRef.current  = 0;
      reconnectAttemptState.current = 0;
      setConnected(true);
      setError(null);
    };

    ws.onmessage = handleMessage;

    ws.onerror = (event) => {
      console.error("[useEEGSocket] Error WS:", event);
      // No llamar setConnected(false) aquí — onclose siempre se dispara
      // después de onerror. Evitar llamadas redundantes al store.
    };

    ws.onclose = (event) => {
      setConnected(false);

      if (!isMountedRef.current) {
        console.log("[useEEGSocket] Conexión cerrada (unmount).");
        return;
      }

      const wasClean    = event.wasClean;
      const attempt     = reconnectAttemptRef.current;

      if (wasClean || attempt >= MAX_ATTEMPTS) {
        if (attempt >= MAX_ATTEMPTS) {
          console.error(
            `[useEEGSocket] ❌ Máximo de intentos (${MAX_ATTEMPTS}) alcanzado. ` +
            `Recarga la página para reconectar.`
          );
        }
        return;
      }

      const delay = getBackoffDelay(attempt);
      reconnectAttemptRef.current++;
      reconnectAttemptState.current = reconnectAttemptRef.current;

      console.log(
        `[useEEGSocket] Reconectando en ${delay / 1000}s ` +
        `(intento ${reconnectAttemptRef.current}/${MAX_ATTEMPTS})...`
      );

      reconnectTimerRef.current = setTimeout(() => {
        if (isMountedRef.current) connect();
      }, delay);
    };
  }, [handleMessage, getBackoffDelay, setConnected, setError]);

  // ── Efecto principal: montar / desmontar ──────────────────────────────
  useEffect(() => {
    isMountedRef.current = true;
    connect();

    // Cerrar WS limpiamente cuando el usuario abandona la página.
    // `beforeunload` es síncrono — WebSocket.close() es inmediato.
    const handleBeforeUnload = () => {
      isMountedRef.current = false; // evitar reconexión automática
      wsRef.current?.close(1000, "Page unload");
    };

    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      // Cleanup al desmontar el componente que usa este hook
      isMountedRef.current = false;

      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }

      // Código 1000 = cierre normal (no dispara reconexión en onclose)
      wsRef.current?.close(1000, "Component unmount");
      wsRef.current = null;

      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  // Evita re-suscripciones WS por cambios de identidad en callbacks durante renders.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── sendMessage ───────────────────────────────────────────────────────
  const sendMessage = useCallback((msg: WSMessage): void => {
    const ws = wsRef.current;

    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn("[useEEGSocket] sendMessage: WS no está OPEN. Mensaje descartado:", msg);
      return;
    }

    try {
      ws.send(JSON.stringify(msg));
    } catch (err) {
      console.error("[useEEGSocket] Error en ws.send:", err);
    }
  }, []);

  // ── API de alto nivel ─────────────────────────────────────────────────

  const startSession = useCallback(
    (config: SessionConfig = {}) => {
      setSessionConfig(config);
      sendMessage({ type: "start_session", payload: config });
    },
    [sendMessage, setSessionConfig]
  );

  const stopSession = useCallback(() => {
    sendMessage({ type: "stop_session" });
  }, [sendMessage]);

  const setTranceMode = useCallback(
    (enabled: boolean) => {
      sendMessage({ type: "set_trance_mode", payload: { enabled } });
    },
    [sendMessage]
  );

  const submitSubjective = useCallback(
    (measure: SubjectiveMeasure) => {
      sendMessage({ type: "submit_subjective", payload: measure });
    },
    [sendMessage]
  );

  const setInspectionChannel = useCallback(
    (channel: string) => {
      setInspectionChannelState(channel);
      sendMessage({ type: "set_inspection_channel", payload: { channel } });
    },
    [sendMessage, setInspectionChannelState]
  );

  const loadDataset = useCallback(
    (filePath: string) => {
      sendMessage({ type: "load_dataset", payload: { path: filePath } });
    },
    [sendMessage]
  );

  const setPlaybackPosition = useCallback(
    (seconds: number) => {
      sendMessage({ type: "set_playback_position", payload: { seconds } });
    },
    [sendMessage]
  );

  // ── Valor de retorno ──────────────────────────────────────────────────
  return {
    isConnected,
    isSessionActive,
    reconnectAttempt : reconnectAttemptState.current,
    sendMessage,
    startSession,
    stopSession,
    setTranceMode,
    submitSubjective,
    setInspectionChannel,
    loadDataset,
    dataset,
    playback,
    setPlaybackPosition,
  };
}
