import { describe, it, expect } from "vitest";
import {
  episodeAgeMin,
  etaEffective,
  flatAgeMin,
  hazardAt,
  priorFor,
  rulForDiag,
  rulQuantile,
  survivalAt,
  weibullQuantile,
} from "../src/lib/rul";
import type { Sample } from "../src/lib/sim";

const mk = (pt: number, pc: number, pl: number, temp: number, q: number, choke: number, m: number): Sample => ({
  m, pt, pc, pl, temp, q, choke,
});

const BASE = { pt: 560, pc: 780, pl: 430, temp: 44, q: 2250, choke: 54 };

describe("rul — núcleo Weibull", () => {
  it("S(0)=1 y S(t) es decreciente con asíntota 0", () => {
    expect(survivalAt(0, 720, 1.8)).toBe(1);
    expect(survivalAt(100, 720, 1.8)).toBeGreaterThan(survivalAt(200, 720, 1.8));
    expect(survivalAt(1e6, 720, 1.8)).toBeLessThan(1e-6);
  });

  it("el cuantil 0.5 es la mediana η·(ln2)^(1/β)", () => {
    expect(weibullQuantile(0.5, 720, 1.8)).toBeCloseTo(720 * Math.pow(Math.LN2, 1 / 1.8), 6);
  });

  it("el cuantil condicional satisface S(t + x_p) = S(t)·(1−p)", () => {
    const eta = 720, beta = 1.8, t = 140;
    for (const p of [0.1, 0.5, 0.9]) {
      const x = rulQuantile(t, p, eta, beta);
      const ratio = survivalAt(t + x, eta, beta) / survivalAt(t, eta, beta);
      expect(ratio).toBeCloseTo(1 - p, 6);
    }
  });

  it("hazard es creciente para β>1 y constante para β=1", () => {
    expect(hazardAt(200, 720, 1.8)).toBeGreaterThan(hazardAt(100, 720, 1.8));
    expect(hazardAt(300, 720, 1)).toBeCloseTo(hazardAt(50, 720, 1), 9);
  });

  it("los cuantiles condicionales están ordenados p10 < mediana < p90", () => {
    const p10 = rulQuantile(60, 0.1, 480, 2.2);
    const p50 = rulQuantile(60, 0.5, 480, 2.2);
    const p90 = rulQuantile(60, 0.9, 480, 2.2);
    expect(p10).toBeLessThan(p50);
    expect(p50).toBeLessThan(p90);
  });
});

describe("rul — aceleración por covariables (AFT)", () => {
  it("η efectiva decrece monótonamente con la severidad", () => {
    const lo = etaEffective(720, { anomScore: 20, diagConf: 0.3, qDropFrac: 0.05 });
    const hi = etaEffective(720, { anomScore: 90, diagConf: 0.9, qDropFrac: 0.4 });
    expect(hi).toBeLessThan(lo);
    const calm = etaEffective(720, { anomScore: 0, diagConf: 0, qDropFrac: 0 });
    expect(calm).toBeCloseTo(720, 6);
  });

  it("priorFor resuelve prefijos sensor-*/spike-* y cae a normal", () => {
    expect(priorFor("sensor-pt").eta).toBe(240);
    expect(priorFor("spike-pl").beta).toBe(3.0);
    expect(priorFor("control-issue").eta).toBe(360);
    expect(priorFor("desconocido")).toEqual(priorFor("normal"));
  });
});

describe("rul — edad de episodio sobre buffers", () => {
  it("cuenta hacia atrás mientras la condición se cumple", () => {
    const buf: Sample[] = [];
    for (let i = 0; i < 40; i++) buf.push(mk(560, 780, 430, 44, 2250, 54, i)); // normal
    for (let i = 40; i < 70; i++) buf.push(mk(470, 780, 430, 44, 2250, 54, i)); // pt cae 90 psi
    const age = episodeAgeMin(buf, "pt", (v) => v < BASE.pt * 0.96);
    expect(age).toBe(30);
  });

  it("no atribuye duración a huecos sin muestras", () => {
    const buf = [
      mk(470, 780, 430, 44, 2250, 54, 10),
      mk(470, 780, 430, 44, 2250, 54, 11),
      mk(470, 780, 430, 44, 2250, 54, 15),
      mk(470, 780, 430, 44, 2250, 54, 16),
    ];
    expect(episodeAgeMin(buf, "pt", (v) => v < 500)).toBe(2);
  });

  it("flatAgeMin detecta un sensor congelado", () => {
    const buf: Sample[] = [];
    for (let i = 0; i < 20; i++)
      buf.push(mk(560 + (i % 3), 780, 430, 44, 2250 + (i % 5), 54, i)); // q con ruido
    for (let i = 20; i < 50; i++)
      buf.push(mk(612, 780, 430, 44, 2250 + (i % 5), 54, i)); // pt congelado
    expect(flatAgeMin(buf, "pt")).toBe(30);
    // el punto final se cuenta a sí mismo: 1 sin congelación real
    expect(flatAgeMin(buf, "q")).toBeLessThanOrEqual(1);
  });
});

describe("rulForDiag — integración", () => {
  it("un liquid loading en desarrollo produce RUL coherente y determinista", () => {
    const buf: Sample[] = [];
    for (let i = 0; i < 100; i++) buf.push(mk(560, 780, 430, 44, 2250, 54, i));
    for (let i = 100; i < 240; i++) {
      const s = Math.min(1, (i - 100) / 140);
      buf.push(mk(560 - 60 * s, 780, 430, 44, 2250 * (1 - 0.12 * s), 54, i));
    }
    const r1 = rulForDiag("liquid-loading", 0.62, buf, BASE, 55);
    const r2 = rulForDiag("liquid-loading", 0.62, buf, BASE, 55);
    expect(r2).toEqual(r1);
    expect(r1.ageH).toBeGreaterThan(0);
    expect(r1.p10).toBeLessThan(r1.rulMedian);
    expect(r1.rulMedian).toBeLessThan(r1.p90);
    expect(r1.survNow).toBeGreaterThan(0);
    expect(r1.survNow).toBeLessThan(1);
    expect(r1.etaEff).toBeLessThan(r1.eta0);
  });

  it("mayor severidad acorta la mediana RUL", () => {
    const buf: Sample[] = [];
    for (let i = 0; i < 120; i++) buf.push(mk(500, 780, 430, 44, 2100, 54, i));
    const mild = rulForDiag("liquid-loading", 0.45, buf, BASE, 35);
    const severe = rulForDiag("liquid-loading", 0.9, buf, BASE, 88);
    expect(severe.rulMedian).toBeLessThan(mild.rulMedian);
    expect(severe.etaEff).toBeLessThan(mild.etaEff);
  });

  it("modo normal: edad 0 y horizonte largo de integridad", () => {
    const buf: Sample[] = [];
    for (let i = 0; i < 80; i++) buf.push(mk(560, 780, 430, 44, 2250, 54, i));
    const r = rulForDiag("normal", 0.8, buf, BASE, 12);
    expect(r.ageH).toBe(0);
    expect(r.rulMedian).toBeCloseTo(r.etaEff * Math.pow(Math.LN2, 1 / r.beta), 6);
    expect(r.rulMedian).toBeGreaterThan(1000);
  });
});
