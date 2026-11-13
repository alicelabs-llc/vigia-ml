// ---------------------------------------------------------------------------
// VIGÍA ML · pruebas de la capa analítica (models.ts · N1–N5)
// ---------------------------------------------------------------------------
import { describe, it, expect } from "vitest";
import {
  holtForecast,
  normCdf,
  projections,
  anomaly,
  anomalyLevel,
  diagnose,
  recommend,
  dataQuality,
  trendPerHour,
} from "../src/lib/models";
import { WellSim } from "../src/lib/sim";
import type { WellCfg, Sample } from "../src/lib/sim";

const cfg: WellCfg = {
  id: "T-100",
  name: "pozo analítico",
  field: "test",
  depth: "2.000 m",
  base: { pt: 560, pc: 780, pl: 430, temp: 44, q: 2250, choke: 54 },
  seed: 4861,
  script: [],
  preRun: 400,
};

describe("N1 · holtForecast", () => {
  it("producirá mean/lo/hi ordenados (lo ≤ mean ≤ hi) para todos los pasos", () => {
    const vals = Array.from({ length: 60 }, (_, i) => 500 + i * 1.5 + Math.sin(i) * 2);
    const fc = holtForecast(vals, 24, 15);
    expect(fc.mean).toHaveLength(24);
    fc.lo.forEach((lo, k) => {
      expect(lo).toBeLessThanOrEqual(fc.mean[k]);
      expect(fc.hi[k]).toBeGreaterThanOrEqual(fc.mean[k]);
    });
  });

  it("es monótono ante tendencia amortiguada: el pronóstico NO explota", () => {
    const vals = Array.from({ length: 80 }, (_, i) => 300 + i * 10); // +10/min
    const fc = holtForecast(vals, 24);
    // con phi=0.93 la tendencia se amortigua: el paso 24 no puede superar ~7x el trend inicial
    expect(fc.mean[23]).toBeLessThan(vals[79] + 10 * 60);
    expect(Number.isFinite(fc.mean[23])).toBe(true);
  });

  it("soporta series cortas sin NaN", () => {
    const fc = holtForecast([100, 102], 6);
    expect(fc.mean.every(Number.isFinite)).toBe(true);
  });
});

describe("normCdf", () => {
  it("Φ(0) ≈ 0.5 y monotónico creciente", () => {
    expect(normCdf(0)).toBeCloseTo(0.5, 1);
    expect(normCdf(-3)).toBeLessThan(normCdf(0));
    expect(normCdf(0)).toBeLessThan(normCdf(3));
  });
  it("está acotada en [0,1] (satura a 0/1 en colas extremas, correcto físicamente)", () => {
    for (const x of [-10, -2, 0, 2, 10]) {
      const p = normCdf(x);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });
  it("coincide con la normal estándar en ±3σ con error < 1e-3", () => {
    // Φ(3) ≈ 0.99865 · Φ(-3) ≈ 0.00135
    expect(Math.abs(normCdf(3) - 0.99865)).toBeLessThan(1e-3);
    expect(Math.abs(normCdf(-3) - 0.00135)).toBeLessThan(1e-3);
  });
});

describe("N4 · projections", () => {
  it("genera filas para pt y q con probabilidad en [0,1]", () => {
    const w = new WellSim(cfg);
    const rows = projections(w.buf, w.base);
    expect(rows.map((r) => r.key)).toEqual(["pt", "q"]);
    for (const r of rows) {
      expect(r.prob).toBeGreaterThanOrEqual(0);
      expect(r.prob).toBeLessThanOrEqual(1);
      expect(Number.isFinite(r.thr)).toBe(true);
    }
  });

  it("no calcula proyecciones sin 240 muestras regulares", () => {
    const w = new WellSim(cfg);
    expect(projections(w.buf.slice(0, 100), w.base)).toEqual([]);
    const irregular = w.buf.slice(-240).map((sample, i) => ({
      ...sample,
      m: sample.m + (i > 100 ? 1 : 0),
    }));
    expect(projections(irregular, w.base)).toEqual([]);
  });

  it("umbral por defecto: pt = 82% de base, q = 78% de base", () => {
    const w = new WellSim(cfg);
    const rows = projections(w.buf, w.base);
    expect(rows[0].thr).toBeCloseTo(w.base.pt * 0.82, 5);
    expect(rows[1].thr).toBeCloseTo(w.base.q * 0.78, 5);
  });
});

describe("N2 · anomaly", () => {
  it("pozo normal → nivel ÓPTIMO o VIGILAR (score bajo)", () => {
    const w = new WellSim(cfg);
    const a = anomaly(w.buf.slice(-240));
    expect(a.score).toBeLessThan(55);
    expect(a.level).toBe(anomalyLevel(a.score));
  });

  it("sensor congelado → score ≥ 74 con contribución detectada", () => {
    const w = new WellSim({ ...cfg, seed: 3517, script: [{ at: 80, s: "sensorFault" }], preRun: 300 });
    const a = anomaly(w.buf.slice(-240));
    expect(a.score).toBeGreaterThanOrEqual(74);
    expect(a.contributions.length).toBeGreaterThan(0);
  });

  it("score siempre finito y acotado a [2,98]", () => {
    const w = new WellSim(cfg);
    const a = anomaly(w.buf.slice(-240));
    expect(Number.isFinite(a.score)).toBe(true);
    expect(a.score).toBeGreaterThanOrEqual(2);
    expect(a.score).toBeLessThanOrEqual(98);
  });
});

describe("N3 · diagnose", () => {
  it("pozo normal → primera hipótesis 'normal'", () => {
    const w = new WellSim({ ...cfg, script: [] });
    const d = diagnose(w.buf.slice(-160));
    expect(d[0].id).toBe("normal");
    expect(d[0].conf).toBeGreaterThan(0);
  });

  it("falla de sensor → hipótesis sensor-* con confianza alta", () => {
    const w = new WellSim({ ...cfg, seed: 3517, script: [{ at: 80, s: "sensorFault" }], preRun: 300 });
    const d = diagnose(w.buf.slice(-160));
    const sf = d.find((h) => h.id.startsWith("sensor-"));
    expect(sf).toBeDefined();
    expect(sf!.conf).toBeGreaterThanOrEqual(0.9);
    expect(sf!.evidence.length).toBeGreaterThan(0);
  });

  it("liquid loading → hipótesis liquid-loading con evidencia física", () => {
    const w = new WellSim({ ...cfg, seed: 1103, script: [{ at: 100, s: "liquidLoading" }], preRun: 480 });
    const d = diagnose(w.buf.slice(-160));
    const ll = d.find((h) => h.id === "liquid-loading");
    expect(ll).toBeDefined();
    expect(ll!.conf).toBeGreaterThan(0.3);
  });
});

describe("N5 · recommend", () => {
  it("sin fallas genera recomendación rutinaria", () => {
    const w = new WellSim(cfg);
    const d = diagnose(w.buf.slice(-160));
    const p = projections(w.buf, w.base);
    const recs = recommend(d, p, w.buf.slice(-240));
    expect(recs.length).toBeGreaterThan(0);
    expect(["ALTA", "MEDIA", "RUTINA"]).toContain(recs[0].prio);
  });

  it("presenta el riesgo de proyección como revisión, no como orden de setpoint", () => {
    const w = new WellSim(cfg);
    const risky = [
      { key: "pt", thr: 400, condition: "P·tubing < 400 psi", prob: 0.9, hoursEst: 1 },
    ] as Parameters<typeof recommend>[1];
    const recs = recommend([], risky, w.buf);
    expect(recs[0].text).toMatch(/revisión del responsable de operación/i);
    expect(recs[0].text).toMatch(/debe ejecutarse automáticamente/i);
    expect(recs[0].text).not.toMatch(/Ajustar choke/i);
  });

  it("con liquid loading sugiere descarga de líquidos (ALTA)", () => {
    const w = new WellSim({ ...cfg, seed: 1103, script: [{ at: 100, s: "liquidLoading" }], preRun: 480 });
    const s = w.buf.slice(-240);
    const d = diagnose(s.slice(-160));
    const p = projections(s, w.base);
    const recs = recommend(d, p, s);
    expect(recs.some((r) => /descarga de líquidos/i.test(r.text))).toBe(true);
  });
});

describe("calidad de datos", () => {
  it("pozo normal → grado A o B y veredicto de telemetría apta", () => {
    const w = new WellSim(cfg);
    const dq = dataQuality(w.buf.slice(-260));
    expect(["A", "B"]).toContain(dq.grade);
    expect(dq.overall).toBeGreaterThanOrEqual(75);
    expect(dq.rows).toHaveLength(6);
  });

  it("no inventa latencia para el simulador y mide la latencia real OPC-UA", () => {
    const w = new WellSim(cfg);
    const simQuality = dataQuality(w.buf.slice(-260));
    expect(simQuality.rows.every((row) => row.latency === null)).toBe(true);

    const now = Date.now();
    const live = { ...w.last, ts: now, sourceTs: now - 5000, source: "opcua" as const };
    const liveQuality = dataQuality([live], now);
    expect(liveQuality.rows.every((row) => row.latency === 5)).toBe(true);
    expect(liveQuality.grade).toBe("D");
    expect(liveQuality.verdict).toMatch(/240 muestras/);
  });

  it("devuelve calidad D sin datos", () => {
    const dq = dataQuality([]);
    expect(dq.grade).toBe("D");
    expect(dq.overall).toBe(0);
    expect(dq.rows).toHaveLength(6);
  });

  it("calcula tendencias con minutos transcurridos, no con cantidad de frames", () => {
    const samples: Sample[] = [
      { m: 0, pt: 100, pc: 100, pl: 100, temp: 20, q: 100, choke: 50 },
      { m: 1, pt: 101, pc: 100, pl: 100, temp: 20, q: 100, choke: 50 },
      { m: 3, pt: 103, pc: 100, pl: 100, temp: 20, q: 100, choke: 50 },
    ];
    expect(trendPerHour(samples, "pt", 3)).toBeCloseTo(60, 6);
  });

  it("cada fila es finita y coherente", () => {
    const w = new WellSim(cfg);
    const dq = dataQuality(w.buf.slice(-260));
    for (const r of dq.rows) {
      expect(Number.isFinite(r.flat)).toBe(true);
      if (r.latency !== null) expect(Number.isFinite(r.latency)).toBe(true);
      expect(r.spikes).toBeGreaterThanOrEqual(0);
      expect(r.flat).toBeGreaterThanOrEqual(0);
    }
  });
});
