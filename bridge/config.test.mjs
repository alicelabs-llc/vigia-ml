import { describe, expect, it } from "vitest";
import { CLIENT_TAGS, normalizeDataValue, REQUIRED_TAGS, validateClientTagMap, validateTagMap } from "./config.mjs";

const complete = {
  pt: "ns=2;s=Well.PT",
  pc: "ns=2;s=Well.PC",
  pl: "ns=2;s=Well.PL",
  temp: "ns=2;s=Well.TEMP",
  q: "ns=2;s=Well.Q",
  choke: "ns=2;s=Well.CHOKE",
};

describe("validateTagMap", () => {
  it("accepts a complete map with standard string NodeIds", () => {
    expect(validateTagMap(complete, { requireAll: true })).toEqual(complete);
  });

  it("accepts standard numeric NodeIds", () => {
    expect(validateTagMap({ pt: "ns=0;i=2258" })).toEqual({ pt: "ns=0;i=2258" });
  });

  it("rejects empty, null, array and primitive maps", () => {
    for (const value of [null, [], "map", 4, {}]) {
      expect(() => validateTagMap(value)).toThrow();
    }
  });

  it("rejects unknown telemetry keys", () => {
    expect(() => validateTagMap({ ...complete, pressure: "ns=2;s=Well.P" })).toThrow(/desconocidas/);
  });

  it("rejects malformed NodeIds and trims valid values", () => {
    expect(() => validateTagMap({ pt: "not-a-node-id" })).toThrow(/NodeId inválido/);
    expect(validateTagMap({ pt: " ns=2;s=Well.PT " })).toEqual({ pt: "ns=2;s=Well.PT" });
  });

  it("validates the browser alias map separately from physical OPC-UA NodeIds", () => {
    expect(validateClientTagMap(CLIENT_TAGS)).toEqual(CLIENT_TAGS);
    expect(() => validateClientTagMap({ ...CLIENT_TAGS, pt: "ns=2;s=Well.PT" })).toThrow(/alias/);
    expect(() => validateClientTagMap({ ...CLIENT_TAGS, extra: "EXTRA" })).toThrow(/desconocidas/);
  });

  it("discards bad-quality and invalid telemetry values", () => {
    expect(normalizeDataValue({ statusCode: { isGood: () => false }, value: { value: 12 } })).toBeNull();
    expect(normalizeDataValue({ value: { value: NaN } })).toBeNull();
    expect(normalizeDataValue({ value: { value: "12" } })).toBeNull();
  });

  it("preserves source timestamps and safely falls back", () => {
    const ts = new Date("2026-11-12T12:00:00.000Z").getTime();
    expect(normalizeDataValue({ value: { value: 12 }, sourceTimestamp: new Date(ts) })).toEqual({ value: 12, ts });
    expect(normalizeDataValue({ value: { value: 12 }, sourceTimestamp: new Date("invalid") }, 123)).toEqual({ value: 12, ts: 123 });
  });

  it("requires all six process variables for a full real-world subscription", () => {
    expect(() => validateTagMap({ pt: complete.pt }, { requireAll: true })).toThrow(/faltan variables/);
    expect(REQUIRED_TAGS).toHaveLength(6);
  });
});
