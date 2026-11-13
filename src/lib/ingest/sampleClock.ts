import { SIM_EPOCH, VAR_KEYS, sanitizeSample } from "../sim";
import type { Sample, VarKey } from "../sim";
import type { IngestSample } from "./opcuaBridge";

export type SourceSampleKind = "initial" | "update" | "new-minute";
export interface SourceSampleUpdate {
  kind: SourceSampleKind;
  sample: Sample;
}

/**
 * Combina notificaciones por variable en una sola muestra por minuto de origen.
 * Las variables que no cambian se conservan como sample-and-hold; nunca se
 * cuenta cada tag como un minuto independiente.
 */
export class SourceSampleAccumulator {
  constructor(private readonly epochMs = SIM_EPOCH) {}

  private values: Partial<Record<VarKey, number>> = {};
  private lastMinute: number | null = null;
  private lastTs = 0;
  private sourceTimestamps: Partial<Record<VarKey, number>> = {};
  private elapsedMin = 0;
  private ready = false;
  private lastSample: Sample | null = null;

  ingest(input: IngestSample): SourceSampleUpdate | null {
    if (!Number.isFinite(input?.ts) || input.ts <= 0) return null;
    const minute = Math.floor(input.ts / 60000);
    if (this.lastMinute !== null && minute < this.lastMinute) return null;

    for (const key of VAR_KEYS) {
      const value = input.values?.[key];
      if (typeof value === "number" && Number.isFinite(value)) {
        this.values[key] = value;
        if (input.timestampOrigin === "source") this.sourceTimestamps[key] = input.ts;
        else delete this.sourceTimestamps[key];
      }
    }

    const previousMinute = this.lastMinute;
    if (!this.ready) {
      this.lastMinute = minute;
      this.lastTs = Math.max(this.lastTs, input.ts);
      if (!VAR_KEYS.every((key) => this.values[key] !== undefined)) return null;
      this.elapsedMin = minute - Math.floor(this.epochMs / 60000);
      this.ready = true;
      return this.makeUpdate("initial");
    }

    if (previousMinute !== null && minute > previousMinute) {
      this.elapsedMin += minute - previousMinute;
    }
    this.lastMinute = minute;
    this.lastTs = Math.max(this.lastTs, input.ts);
    return this.makeUpdate(previousMinute !== null && minute > previousMinute ? "new-minute" : "update");
  }

  current(): Sample | null {
    return this.lastSample ? { ...this.lastSample } : null;
  }

  private makeUpdate(kind: SourceSampleKind): SourceSampleUpdate {
    const sample = sanitizeSample({
      m: this.elapsedMin,
      ts: this.lastTs,
      source: "opcua",
      sourceTs: VAR_KEYS.every((key) => Number.isFinite(this.sourceTimestamps[key]))
        ? Math.min(...VAR_KEYS.map((key) => this.sourceTimestamps[key]!))
        : undefined,
      pt: this.values.pt!,
      pc: this.values.pc!,
      pl: this.values.pl!,
      temp: this.values.temp!,
      q: this.values.q!,
      choke: this.values.choke!,
    });
    this.lastSample = sample;
    return { kind, sample };
  }
}
