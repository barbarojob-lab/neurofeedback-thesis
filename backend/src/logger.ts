/**
 * logger.ts  —  backend/src/
 *
 * Sistema de logging estructurado con niveles de severidad.
 *
 * Propósito:
 *  - Reemplazar console.log/error dispersos con una API consistente
 *  - Facilitar filtrado por nivel (debug, info, warn, error, fatal)
 *  - Información temporal estructurada para análisis de sesiones
 *
 * Uso:
 *   import { logger } from "./logger";
 *   logger.info("Sesión iniciada", { sessionId: "s123", patientId: "P001" });
 *   logger.error("Fallo en FFT", { epoch: 42, reason: "beta = 0" });
 */

export type LogLevel = "debug" | "info" | "warn" | "error" | "fatal";

interface LogEntry {
  timestamp: string;  // ISO 8601
  level: LogLevel;
  message: string;
  meta?: Record<string, any>;
  stack?: string;     // para errores
}

class Logger {
  private minLevel: LogLevel = "info";
  private readonly levelOrder: { [key in LogLevel]: number } = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
    fatal: 4,
  };

  constructor(minLevel: LogLevel = "info") {
    this.minLevel = minLevel;
  }

  private shouldLog(level: LogLevel): boolean {
    return this.levelOrder[level] >= this.levelOrder[this.minLevel];
  }

  private format(entry: LogEntry): string {
    const { timestamp, level, message, meta, stack } = entry;
    const levelUpper = level.toUpperCase().padEnd(5);
    const metaStr = meta ? " " + JSON.stringify(meta) : "";
    const stackStr = stack ? `\n${stack}` : "";
    return `[${timestamp}] [${levelUpper}] ${message}${metaStr}${stackStr}`;
  }

  debug(message: string, meta?: Record<string, any>): void {
    if (!this.shouldLog("debug")) return;
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level: "debug",
      message,
      ...(meta && { meta }),
    };
    console.log(this.format(entry));
  }

  info(message: string, meta?: Record<string, any>): void {
    if (!this.shouldLog("info")) return;
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level: "info",
      message,
      ...(meta && { meta }),
    };
    console.log(this.format(entry));
  }

  warn(message: string, meta?: Record<string, any>): void {
    if (!this.shouldLog("warn")) return;
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level: "warn",
      message,
      ...(meta && { meta }),
    };
    console.warn(this.format(entry));
  }

  error(message: string, meta?: Record<string, any>, error?: Error): void {
    if (!this.shouldLog("error")) return;
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level: "error",
      message,
      ...(meta && { meta }),
      ...(error?.stack && { stack: error.stack }),
    };
    console.error(this.format(entry));
  }

  fatal(message: string, meta?: Record<string, any>, error?: Error): void {
    if (!this.shouldLog("fatal")) return;
    const entry: LogEntry = {
      timestamp: new Date().toISOString(),
      level: "fatal",
      message,
      ...(meta && { meta }),
      ...(error?.stack && { stack: error.stack }),
    };
    console.error(this.format(entry));
    // En producción, podría enviar a servicio de alertas
  }

  /**
   * Cambiar el nivel mínimo de log (e.g., en desarrollo= debug, prod=warn)
   */
  setMinLevel(level: LogLevel): void {
    this.minLevel = level;
  }
}

// Singleton exportado
export const logger = new Logger(
  (process.env.LOG_LEVEL as LogLevel) ?? "info"
);
