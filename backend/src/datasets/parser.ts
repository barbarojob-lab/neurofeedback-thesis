import fs from "fs/promises";
import path from "path";

export interface DatasetMetadata {
  id: string;
  sourcePath: string;
  format: "edf" | "csv";
  channels: string[];
  sampleRate: number;
  durationSec: number;
  totalSamples: number;
}

async function parseEdfMetadata(filePath: string): Promise<DatasetMetadata> {
  const stat = await fs.stat(filePath);
  if (!stat.isFile()) {
    throw new Error("La ruta no es un archivo");
  }

  const fd = await fs.open(filePath, "r");
  try {
    const fixedHeader = Buffer.alloc(256);
    await fd.read(fixedHeader, 0, 256, 0);

    const numDataRecords = Number.parseInt(fixedHeader.toString("ascii", 236, 244).trim(), 10);
    const dataRecordDurationSec = Number.parseFloat(fixedHeader.toString("ascii", 244, 252).trim());
    const ns = Number.parseInt(fixedHeader.toString("ascii", 252, 256).trim(), 10);

    if (!Number.isFinite(ns) || ns <= 0) {
      throw new Error("EDF inválido: número de señales (ns) no válido");
    }

    const variableHeader = Buffer.alloc(ns * 256);
    await fd.read(variableHeader, 0, ns * 256, 256);

    // EDF almacena etiquetas de canal en bloques de 16 bytes por señal.
    const channels: string[] = [];
    for (let i = 0; i < ns; i++) {
      const start = i * 16;
      const label = variableHeader.toString("ascii", start, start + 16).trim();
      channels.push(label || `ch_${i + 1}`);
    }

    // samplesPerRecord por canal está en el bloque 9 de la cabecera variable.
    const samplesPerRecordOffset = ns * (16 + 80 + 8 + 8 + 8 + 8 + 80);
    const samplesPerRecordByChannel: number[] = [];
    for (let i = 0; i < ns; i++) {
      const start = samplesPerRecordOffset + i * 8;
      const value = Number.parseInt(variableHeader.toString("ascii", start, start + 8).trim(), 10);
      samplesPerRecordByChannel.push(Number.isFinite(value) && value > 0 ? value : 0);
    }

    const maxSamplesPerRecord = Math.max(...samplesPerRecordByChannel, 0);
    if (maxSamplesPerRecord <= 0 || !Number.isFinite(dataRecordDurationSec) || dataRecordDurationSec <= 0) {
      throw new Error("EDF inválido: no se pudo inferir sample rate");
    }

    const sampleRate = Math.round(maxSamplesPerRecord / dataRecordDurationSec);
    const durationSec = Number.isFinite(numDataRecords) && numDataRecords > 0
      ? numDataRecords * dataRecordDurationSec
      : 0;
    const totalSamples = Math.max(0, Math.round(durationSec * sampleRate));

    return {
      id: `${path.basename(filePath)}_${Date.now()}`,
      sourcePath: filePath,
      format: "edf",
      channels,
      sampleRate,
      durationSec,
      totalSamples,
    };
  } finally {
    await fd.close();
  }
}

async function parseCsvMetadata(filePath: string): Promise<DatasetMetadata> {
  const content = await fs.readFile(filePath, "utf8");
  const [headerLine, ...rows] = content.split(/\r?\n/).filter(Boolean);

  if (!headerLine) {
    throw new Error("CSV vacío");
  }

  const columns = headerLine.split(",").map((s) => s.trim());
  const channels = columns.filter((c) => c.toLowerCase() !== "timestamp");

  // Si hay timestamp en ms, estimamos fs; si no, usamos fallback 250.
  let sampleRate = 250;
  const timestampIdx = columns.findIndex((c) => c.toLowerCase() === "timestamp");
  if (timestampIdx >= 0 && rows.length > 3) {
    const t0 = Number(rows[0].split(",")[timestampIdx]);
    const t1 = Number(rows[1].split(",")[timestampIdx]);
    const dt = t1 - t0;
    if (Number.isFinite(dt) && dt > 0) {
      sampleRate = Math.round(1000 / dt);
    }
  }

  const totalSamples = rows.length;
  const durationSec = totalSamples / Math.max(sampleRate, 1);

  return {
    id: `${path.basename(filePath)}_${Date.now()}`,
    sourcePath: filePath,
    format: "csv",
    channels,
    sampleRate,
    durationSec,
    totalSamples,
  };
}

export async function parseDatasetMetadata(filePath: string): Promise<DatasetMetadata> {
  const ext = path.extname(filePath).toLowerCase();

  if (ext === ".edf") {
    return parseEdfMetadata(filePath);
  }

  if (ext === ".csv") {
    return parseCsvMetadata(filePath);
  }

  throw new Error(`Formato no soportado: ${ext}. Usa .edf o .csv`);
}
