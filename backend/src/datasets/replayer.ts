import fs from "fs/promises";
import path from "path";

import type { DatasetMetadata } from "./parser";

interface LoadedDataset<TChannel extends string> {
  metadata: DatasetMetadata;
  channels: Record<TChannel, Float32Array>;
  sampleCount: number;
}

interface ChannelDecodeState {
  acc: number;
  out: number[];
}

function normalizeLabel(label: string): string {
  return label
    .trim()
    .toUpperCase()
    .replace(/[\s_\-]+/g, "")
    .replace(/^EEG/, "");
}

function parseAsciiInt(buf: Buffer, start: number, end: number): number {
  return Number.parseInt(buf.toString("ascii", start, end).trim(), 10);
}

function parseAsciiFloat(buf: Buffer, start: number, end: number): number {
  return Number.parseFloat(buf.toString("ascii", start, end).trim());
}

function parseEdfHeader(raw: Buffer) {
  if (raw.length < 256) {
    throw new Error("EDF invalido: cabecera incompleta");
  }

  const headerBytes = parseAsciiInt(raw, 184, 192);
  const numRecords = parseAsciiInt(raw, 236, 244);
  const recordDurationSec = parseAsciiFloat(raw, 244, 252);
  const ns = parseAsciiInt(raw, 252, 256);

  if (!Number.isFinite(headerBytes) || headerBytes <= 0) {
    throw new Error("EDF invalido: header bytes no valido");
  }
  if (!Number.isFinite(numRecords) || numRecords <= 0) {
    throw new Error("EDF invalido: numero de data records no valido");
  }
  if (!Number.isFinite(recordDurationSec) || recordDurationSec <= 0) {
    throw new Error("EDF invalido: duracion de data record no valida");
  }
  if (!Number.isFinite(ns) || ns <= 0) {
    throw new Error("EDF invalido: numero de senales no valido");
  }
  if (raw.length < headerBytes) {
    throw new Error("EDF invalido: archivo truncado antes del header completo");
  }

  const variableHeader = raw.subarray(256, headerBytes);

  const labels: string[] = [];
  const physicalMins: number[] = [];
  const physicalMaxs: number[] = [];
  const digitalMins: number[] = [];
  const digitalMaxs: number[] = [];
  const samplesPerRecord: number[] = [];

  let offset = 0;
  for (let i = 0; i < ns; i++) {
    labels.push(variableHeader.toString("ascii", offset, offset + 16).trim());
    offset += 16;
  }

  offset += ns * 80; // transducer
  offset += ns * 8;  // physical dimension

  for (let i = 0; i < ns; i++) {
    physicalMins.push(parseAsciiFloat(variableHeader, offset, offset + 8));
    offset += 8;
  }

  for (let i = 0; i < ns; i++) {
    physicalMaxs.push(parseAsciiFloat(variableHeader, offset, offset + 8));
    offset += 8;
  }

  for (let i = 0; i < ns; i++) {
    digitalMins.push(parseAsciiFloat(variableHeader, offset, offset + 8));
    offset += 8;
  }

  for (let i = 0; i < ns; i++) {
    digitalMaxs.push(parseAsciiFloat(variableHeader, offset, offset + 8));
    offset += 8;
  }

  offset += ns * 80; // prefiltering

  for (let i = 0; i < ns; i++) {
    samplesPerRecord.push(parseAsciiInt(variableHeader, offset, offset + 8));
    offset += 8;
  }

  return {
    headerBytes,
    numRecords,
    recordDurationSec,
    ns,
    labels,
    physicalMins,
    physicalMaxs,
    digitalMins,
    digitalMaxs,
    samplesPerRecord,
  };
}

export class DatasetReplayer<TChannel extends string> {
  private readonly targetChannels: readonly TChannel[];
  private readonly targetSampleRate: number;

  private loaded: LoadedDataset<TChannel> | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private cursor = 0;

  constructor(targetSampleRate: number, targetChannels: readonly TChannel[]) {
    this.targetSampleRate = targetSampleRate;
    this.targetChannels = targetChannels;
  }

  isLoaded(): boolean {
    return this.loaded !== null;
  }

  reset(): void {
    this.cursor = 0;
  }

  seekToSeconds(seconds: number): { positionSec: number; durationSec: number } {
    if (!this.loaded) {
      throw new Error("No hay dataset cargado");
    }

    const total = this.loaded.sampleCount;
    if (total <= 0) {
      this.cursor = 0;
      return { positionSec: 0, durationSec: 0 };
    }

    const targetSample = Math.round(Math.max(0, Number(seconds) || 0) * this.targetSampleRate);
    this.cursor = Math.min(Math.max(0, targetSample), Math.max(0, total - 1));

    return {
      positionSec: this.cursor / this.targetSampleRate,
      durationSec: total / this.targetSampleRate,
    };
  }

  getPlaybackInfo(): { positionSec: number; durationSec: number } | null {
    if (!this.loaded) return null;
    return {
      positionSec: this.cursor / this.targetSampleRate,
      durationSec: this.loaded.sampleCount / this.targetSampleRate,
    };
  }

  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  async load(filePath: string): Promise<DatasetMetadata> {
    const ext = path.extname(filePath).toLowerCase();

    if (ext === ".csv") {
      this.loaded = await this.loadCsv(filePath);
      return this.loaded.metadata;
    }

    if (ext === ".edf") {
      this.loaded = await this.loadEdf(filePath);
      return this.loaded.metadata;
    }

    throw new Error(`Formato no soportado para replay: ${ext}. Usa .edf o .csv`);
  }

  start(onSample: (sample: Record<TChannel, number>) => void): void {
    if (!this.loaded) {
      throw new Error("No hay dataset cargado. Usa load_dataset primero.");
    }
    if (this.intervalId) {
      return;
    }

    const intervalMs = 1000 / this.targetSampleRate;

    this.intervalId = setInterval(() => {
      if (!this.loaded || this.loaded.sampleCount <= 0) {
        return;
      }

      if (this.cursor >= this.loaded.sampleCount) {
        this.cursor = 0; // loop continuo para sesiones largas
      }

      const frame = {} as Record<TChannel, number>;
      for (const ch of this.targetChannels) {
        frame[ch] = this.loaded.channels[ch][this.cursor] ?? 0;
      }

      onSample(frame);
      this.cursor += 1;
    }, intervalMs);
  }

  private async loadCsv(filePath: string): Promise<LoadedDataset<TChannel>> {
    const raw = await fs.readFile(filePath, "utf8");
    const lines = raw.split(/\r?\n/).filter((line) => line.trim().length > 0);
    if (lines.length < 2) {
      throw new Error("CSV invalido: sin filas de datos");
    }

    const firstChannel = this.targetChannels[0];
    if (!firstChannel) {
      throw new Error("No hay canales objetivo configurados");
    }

    const headerLine = lines[0];
    if (!headerLine) {
      throw new Error("CSV invalido: cabecera ausente");
    }
    const header = headerLine.split(",").map((c) => c.trim());
    const normalizedToIdx = new Map<string, number>();
    for (let i = 0; i < header.length; i++) {
      normalizedToIdx.set(normalizeLabel(header[i] ?? ""), i);
    }

    const colByChannel = new Map<TChannel, number>();
    for (const ch of this.targetChannels) {
      const idx = normalizedToIdx.get(normalizeLabel(ch));
      if (idx === undefined) {
        throw new Error(`CSV no contiene canal requerido: ${ch}`);
      }
      colByChannel.set(ch, idx);
    }

    const tsIdx = normalizedToIdx.get("TIMESTAMP");
    let sourceFs = this.targetSampleRate;

    if (tsIdx !== undefined && lines.length >= 3) {
      const row1 = lines[1];
      const row2 = lines[2];
      if (!row1 || !row2) {
        throw new Error("CSV invalido: filas insuficientes para estimar sample rate");
      }
      const t0 = Number(row1.split(",")[tsIdx]);
      const t1 = Number(row2.split(",")[tsIdx]);
      const dtMs = t1 - t0;
      if (Number.isFinite(dtMs) && dtMs > 0) {
        sourceFs = Math.round(1000 / dtMs);
      }
    }

    const byChannel = this.targetChannels.reduce((acc, ch) => {
      acc[ch] = [] as number[];
      return acc;
    }, {} as Record<TChannel, number[]>);

    for (let row = 1; row < lines.length; row++) {
      const rowLine = lines[row];
      if (!rowLine) continue;
      const cols = rowLine.split(",");
      for (const ch of this.targetChannels) {
        const idx = colByChannel.get(ch)!;
        const value = Number(cols[idx]);
        byChannel[ch].push(Number.isFinite(value) ? value : 0);
      }
    }

    const resampled = this.resampleToTarget(byChannel, sourceFs);
    const sampleCount = resampled[firstChannel]?.length ?? 0;

    return {
      metadata: {
        id: `${path.basename(filePath)}_${Date.now()}`,
        sourcePath: filePath,
        format: "csv",
        channels: [...this.targetChannels],
        sampleRate: this.targetSampleRate,
        durationSec: sampleCount / this.targetSampleRate,
        totalSamples: sampleCount,
      },
      channels: resampled,
      sampleCount,
    };
  }

  private async loadEdf(filePath: string): Promise<LoadedDataset<TChannel>> {
    const raw = await fs.readFile(filePath);
    const hdr = parseEdfHeader(raw);

    const sourceFs = Math.round(
      Math.max(...hdr.samplesPerRecord) / hdr.recordDurationSec
    );

    if (!Number.isFinite(sourceFs) || sourceFs <= 0) {
      throw new Error("EDF invalido: no se pudo inferir sample rate");
    }

    const labelToSignalIdx = new Map<string, number>();
    for (let i = 0; i < hdr.labels.length; i++) {
      labelToSignalIdx.set(normalizeLabel(hdr.labels[i] ?? ""), i);
    }

    const signalIdxByChannel = new Map<TChannel, number>();
    for (const ch of this.targetChannels) {
      const signalIdx = labelToSignalIdx.get(normalizeLabel(ch));
      if (signalIdx === undefined) {
        throw new Error(`EDF no contiene canal requerido: ${ch}`);
      }
      signalIdxByChannel.set(ch, signalIdx);
    }

    const decodeStateBySignalIdx = new Map<number, ChannelDecodeState>();
    for (const ch of this.targetChannels) {
      decodeStateBySignalIdx.set(signalIdxByChannel.get(ch)!, { acc: 0, out: [] });
    }

    const totalRecordBytes = hdr.samplesPerRecord.reduce((sum, n) => sum + n * 2, 0);
    const ratio = this.targetSampleRate / sourceFs;

    let recordOffset = hdr.headerBytes;
    for (let r = 0; r < hdr.numRecords; r++) {
      let channelOffset = recordOffset;

      for (let sigIdx = 0; sigIdx < hdr.ns; sigIdx++) {
        const spr = hdr.samplesPerRecord[sigIdx] ?? 0;
        const decodeState = decodeStateBySignalIdx.get(sigIdx);

        if (decodeState) {
          const physMin = hdr.physicalMins[sigIdx] ?? -32768;
          const physMax = hdr.physicalMaxs[sigIdx] ?? 32767;
          const digMin = hdr.digitalMins[sigIdx] ?? -32768;
          const digMax = hdr.digitalMaxs[sigIdx] ?? 32767;
          const scale = digMax !== digMin ? (physMax - physMin) / (digMax - digMin) : 1;

          for (let i = 0; i < spr; i++) {
            const rawInt = raw.readInt16LE(channelOffset + i * 2);
            const value = (rawInt - digMin) * scale + physMin;

            decodeState.acc += ratio;
            if (decodeState.acc >= 1) {
              decodeState.out.push(value);
              decodeState.acc -= 1;
            }
          }
        }

        channelOffset += spr * 2;
      }

      recordOffset += totalRecordBytes;
      if (recordOffset > raw.length) {
        break;
      }
    }

    const numericByChannel = this.targetChannels.reduce((acc, ch) => {
      const signalIdx = signalIdxByChannel.get(ch)!;
      const state = decodeStateBySignalIdx.get(signalIdx)!;
      acc[ch] = state.out;
      return acc;
    }, {} as Record<TChannel, number[]>);

    const channels = this.toFloat32Aligned(numericByChannel);
    const firstChannel = this.targetChannels[0];
    if (!firstChannel) {
      throw new Error("No hay canales objetivo configurados");
    }
    const sampleCount = channels[firstChannel]?.length ?? 0;

    return {
      metadata: {
        id: `${path.basename(filePath)}_${Date.now()}`,
        sourcePath: filePath,
        format: "edf",
        channels: [...this.targetChannels],
        sampleRate: this.targetSampleRate,
        durationSec: sampleCount / this.targetSampleRate,
        totalSamples: sampleCount,
      },
      channels,
      sampleCount,
    };
  }

  private resampleToTarget(
    byChannel: Record<TChannel, number[]>,
    sourceFs: number
  ): Record<TChannel, Float32Array> {
    if (sourceFs === this.targetSampleRate) {
      return this.toFloat32Aligned(byChannel);
    }

    const ratio = this.targetSampleRate / sourceFs;
    const out = {} as Record<TChannel, number[]>;

    for (const ch of this.targetChannels) {
      const src = byChannel[ch] ?? [];
      const dst: number[] = [];
      let acc = 0;

      for (let i = 0; i < src.length; i++) {
        acc += ratio;
        if (acc >= 1) {
          dst.push(src[i] ?? 0);
          acc -= 1;
        }
      }

      out[ch] = dst;
    }

    return this.toFloat32Aligned(out);
  }

  private toFloat32Aligned(byChannel: Record<TChannel, number[]>): Record<TChannel, Float32Array> {
    let minLen = Number.POSITIVE_INFINITY;
    for (const ch of this.targetChannels) {
      minLen = Math.min(minLen, byChannel[ch]?.length ?? 0);
    }

    if (!Number.isFinite(minLen) || minLen <= 0) {
      throw new Error("No se encontraron muestras utiles en el dataset");
    }

    const out = {} as Record<TChannel, Float32Array>;
    for (const ch of this.targetChannels) {
      out[ch] = Float32Array.from((byChannel[ch] ?? []).slice(0, minLen));
    }

    return out;
  }
}
