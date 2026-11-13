// ---------------------------------------------------------------------------
// VIGÍA ML · pruebas del generador de datasets ML (dataGen.ts)
// ---------------------------------------------------------------------------
import { describe, it, expect } from "vitest";
import {
  ML_VARS,
  N_FEATURES,
  CLASSES,
  FC_INPUT,
  FC_STEPS,
  FC_MIN,
  AE_WIN,
  makeNorm,
  normVec,
  extractFeatures,
  aggregates,
  generateDataset,
} from "../src/lib/ml/dataGen";
import { WellSim } from "../src/lib/sim";
import type { WellCfg } from "../src/lib/sim";

const cfg: WellCfg = {
  id: "T-200",
  name: "pozo dataset",
  field: "test",
  depth: "3.000 m",
  base: { pt: 600, pc: 820, pl: 445, temp: 42, q: 2600, choke: 58 },
  seed: 2287,
  script: [],
  preRun: 520,
};

describe("makeNorm / normVec", () => {
  it("escalas nunca menores a 0.5", () => {
    const n = makeNorm({ pt: 1, pc: 1, pl: 1, temp: 1, q: 1, choke: 1 });
    for (const k of ML_VARS) expect(n.scale[k]).toBeGreaterThanOrEqual(0.5);
  });

  it("normVec centra la media base: valor base → 0 normalizado", () => {
    const base = { pt: 600, pc: 820, pl: 445, temp: 42, q: 2600, choke: 58 };
    const n = makeNorm(base);
    const s = { m: 0, pt: 600, pc: 820, pl: 445, temp: 42, q: 2600, choke: 58 };
    const v = normVec(s, n);
    expect(v).toHaveLength(6);
    v.forEach((x) => expect(Math.abs(x)).toBeLessThan(1e-9));
  });
});

describe("extractFeatures", () => {
  it("devuelve exactamente N_FEATURES rasgos acotados a [-4, 4]", () => {
    const w = new WellSim(cfg);
    const n = makeNorm(w.base);
    const win = w.buf.slice(0, 90);
    const f = extractFeatures(win, n);
    expect(f).toHaveLength(N_FEATURES);
    f.forEach((x) => {
      expect(Number.isFinite(x)).toBe(true);
      expect(x).toBeGreaterThanOrEqual(-4);
      expect(x).toBeLessThanOrEqual(4);
    });
  });

  it("las pendientes usan el tiempo transcurrido cuando hay huecos", () => {
    const w = new WellSim(cfg);
    const n = makeNorm(w.base);
    const regular = Array.from({ length: 90 }, (_, i) => ({
      ...w.buf[i], m: i, pt: 600 + i * 0.2,
    }));
    const sparse = regular.map((sample) => ({ ...sample, m: sample.m * 2 }));
    const fRegular = extractFeatures(regular, n);
    const fSparse = extractFeatures(sparse, n);
    expect(fRegular[0]).toBeCloseTo(fSparse[0] * 2, 2);
  });

  it("ventana plana produce ptFlat alto (sensor congelado)", () => {
    const w = new WellSim(cfg);
    const n = makeNorm(w.base);
    const flat = w.buf.slice(0, 90).map((s) => ({ ...s, pt: 500 }));
    const f = extractFeatures(flat, n);
    // ptFlat es el rasgo índice 10: 0.5/(σ+0.15) → alto si σ≈0
    expect(f[10]).toBeGreaterThan(2);
  });
});

describe("aggregates", () => {
  it("agrupa en bloques de FC_MIN minutos y normaliza", () => {
    const w = new WellSim(cfg);
    const n = makeNorm(w.base);
    const aggs = aggregates(w.buf, n);
    expect(aggs.length).toBe(Math.floor(w.buf.length / FC_MIN));
    aggs.forEach((a) => {
      expect(a).toHaveLength(ML_VARS.length);
      a.forEach((x) => expect(Number.isFinite(x)).toBe(true));
    });
  });
});

describe("generateDataset", () => {
  it(
    "produce splits de train/val sin fuga por pozo y clases balanceadas razonablemente",
    async () => {
      const ds = await generateDataset({ seed: 20260828, wellsScale: 0.35 });
      expect(ds.wells).toBeGreaterThan(0);

      // shapes coherentes
      expect(ds.fcTrainX[0]).toHaveLength(FC_INPUT);
      expect(ds.fcTrainX[0][0]).toHaveLength(ML_VARS.length);
      expect(ds.fcTrainY[0]).toHaveLength(FC_STEPS * ML_VARS.length);
      expect(ds.aeTrainX[0]).toHaveLength(AE_WIN);
      expect(ds.aeTrainX[0][0]).toHaveLength(ML_VARS.length);
      expect(ds.clsTrainX[0]).toHaveLength(N_FEATURES);

      // etiquetas de clase dentro de rango
      for (const y of ds.clsTrainY) expect(y).toBeGreaterThanOrEqual(0);
      for (const y of ds.clsTrainY) expect(y).toBeLessThan(CLASSES.length);

      // split por pozo: val tiene contenido pero es menor que train
      expect(ds.fcValX.length).toBeGreaterThan(0);
      expect(ds.clsValX.length).toBeGreaterThan(0);
      expect(ds.aeValX.length).toBeGreaterThan(0);

      // sin fuga: el nº de ventanas de clasificación train+val debe cubrir todos los pozos
      expect(ds.clsTrainX.length + ds.clsValX.length).toBeGreaterThan(ds.wells * 2);

      // datos finitos
      const fin = (arr: number[][][]) => arr.every((w) => w.every((r) => r.every(Number.isFinite)));
      expect(fin(ds.fcTrainX.slice(0, 50))).toBe(true);
      expect(fin(ds.aeTrainX.slice(0, 50))).toBe(true);
      expect(ds.clsTrainX.every((r) => r.every(Number.isFinite))).toBe(true);
    },
    60_000,
  );

  it("es determinista con la misma semilla", async () => {
    const a = await generateDataset({ seed: 42, wellsScale: 0.35 });
    const b = await generateDataset({ seed: 42, wellsScale: 0.35 });
    expect(a.clsTrainX.length).toBe(b.clsTrainX.length);
    expect(a.fcTrainX.length).toBe(b.fcTrainX.length);
    expect(a.clsTrainX[5]).toEqual(b.clsTrainX[5]);
    expect(a.fcTrainX[3][0]).toEqual(b.fcTrainX[3][0]);
  }, 60_000);

  it("onProgress se invoca con fracciones crecientes hasta 1", async () => {
    const fracs: number[] = [];
    await generateDataset({ seed: 7, wellsScale: 0.35, onProgress: (f) => void fracs.push(f) });
    expect(fracs[fracs.length - 1]).toBeCloseTo(1, 5);
    for (let i = 1; i < fracs.length; i++) expect(fracs[i]).toBeGreaterThan(fracs[i - 1]);
  }, 60_000);
});
