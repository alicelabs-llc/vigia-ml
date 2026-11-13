import { describe, expect, it } from "vitest";
import { SourceSampleAccumulator } from "./sampleClock";
import type { IngestSample } from "./opcuaBridge";

const base = { pt: 560, pc: 780, pl: 430, temp: 44, q: 2250, choke: 54 };
const frame = (ts: number, values: Partial<typeof base>, timestampOrigin: "source" | "receive" = "receive"): IngestSample => ({ ts, values, timestampOrigin });

describe("SourceSampleAccumulator", () => {
  it("waits for the six initial tags before emitting a real sample", () => {
    const acc = new SourceSampleAccumulator(60_000);
    expect(acc.ingest(frame(60_000, { pt: 560 }))).toBeNull();
    expect(acc.ingest(frame(61_000, { pc: 780, pl: 430 }))).toBeNull();
    expect(acc.ingest(frame(62_000, { temp: 44, q: 2250 }))).toBeNull();
    const result = acc.ingest(frame(63_000, { choke: 54 }));
    expect(result?.kind).toBe("initial");
    expect(result?.sample.m).toBe(0);
    expect(result?.sample.source).toBe("opcua");
    expect(result?.sample.pt).toBe(560);
    expect(result?.sample.choke).toBe(54);
  });

  it("retains source timestamps only when every tag has a trusted source timestamp", () => {
    const acc = new SourceSampleAccumulator(60_000);
    for (const key of Object.keys(base) as (keyof typeof base)[]) {
      acc.ingest(frame(60_000, { [key]: base[key] }, "source"));
    }
    expect(acc.current()?.sourceTs).toBe(60_000);
    acc.ingest(frame(75_000, { pt: 555 }, "receive"));
    expect(acc.current()?.sourceTs).toBeUndefined();
  });

  it("retains source timestamps only when every tag has a trusted source timestamp", () => {
    const acc = new SourceSampleAccumulator(60_000);
    for (const key of Object.keys(base) as (keyof typeof base)[]) {
      acc.ingest(frame(60_000, { [key]: base[key] }, "source"));
    }
    expect(acc.current()?.sourceTs).toBe(60_000);
    acc.ingest(frame(75_000, { pt: 555 }, "receive"));
    expect(acc.current()?.sourceTs).toBeUndefined();
  });

  it("merges multiple tag notifications within the same minute", () => {
    const acc = new SourceSampleAccumulator(60_000);
    for (const key of Object.keys(base) as (keyof typeof base)[]) acc.ingest(frame(60_000, { [key]: base[key] }));
    const update = acc.ingest(frame(75_000, { pt: 555 }));
    expect(update?.kind).toBe("update");
    expect(update?.sample.m).toBe(0);
    expect(update?.sample.pt).toBe(555);
    expect(update?.sample.q).toBe(base.q);
  });

  it("advances elapsed minutes by source time, not by frame count", () => {
    const acc = new SourceSampleAccumulator(60_000);
    for (const key of Object.keys(base) as (keyof typeof base)[]) acc.ingest(frame(60_000, { [key]: base[key] }));
    const next = acc.ingest(frame(120_000, { q: 2200 }));
    expect(next?.kind).toBe("new-minute");
    expect(next?.sample.m).toBe(1);
    const gap = acc.ingest(frame(300_000, { pt: 540 }));
    expect(gap?.sample.m).toBe(4);
  });

  it("ignores out-of-order minutes and invalid timestamps", () => {
    const acc = new SourceSampleAccumulator(60_000);
    for (const key of Object.keys(base) as (keyof typeof base)[]) acc.ingest(frame(120_000, { [key]: base[key] }));
    expect(acc.ingest(frame(60_000, { pt: 500 }))).toBeNull();
    expect(acc.ingest(frame(Number.NaN, { pt: 500 }))).toBeNull();
    expect(acc.current()?.pt).toBe(560);
  });

  it("does not emit until every variable has a finite initial value", () => {
    const acc = new SourceSampleAccumulator(60_000);
    expect(acc.ingest(frame(60_000, { pt: 560, pc: 780, pl: 430, temp: 44, q: 2250 }))).toBeNull();
    expect(acc.ingest(frame(61_000, { choke: Number.NaN }))).toBeNull();
    expect(acc.ingest(frame(62_000, { choke: 54 }))?.kind).toBe("initial");
  });
});
