// ---------------------------------------------------------------------------
// VIGÍA · vida útil restante (RUL) por modelo de supervivencia Weibull
// S(t) = exp(−(t/η)^β) · η = vida característica (h) · β = forma
// Priors por modo de falla + modelo AFT (accelerated failure time): las
// covariables operativas (score de anomalía, confianza del diagnóstico y
// caída de caudal) aceleran la falla vía η_ef = η0 · exp(−Σ wᵢ·xᵢ).
// Cuantiles condicionales dado sobrevivir a la edad t del episodio:
//     S(t + x) = S(t)·(1−p)  ⇒  x_p = η·( (t/η)^β − ln(1−p) )^(1/β) − t
// En producción: re-ajustar η0/β por MLE con el historial de fallas real del
// activo (los priors aquí son de literatura y degradan con elegancia).
// Puro, sin DOM: testeable.
// ---------------------------------------------------------------------------

import type { Sample, VarKey } from "./sim";

export interface WeibullPrior {
  eta: number; // vida característica en horas
  beta: number; // exponente de forma (β>1 = desgaste; β≈1 = azar)
}

/**
 * Priors por modo de diagnóstico (literatura de integridad en instalaciones
 * de gas dulce; η en horas de episodio hasta intervención).
 */
export const MODE_PRIORS: Record<string, WeibullPrior> = {
  "liquid-loading": { eta: 720, beta: 1.8 },
  restriction: { eta: 480, beta: 2.2 },
  "control-issue": { eta: 360, beta: 2.6 },
  sensor: { eta: 240, beta: 3.0 },
  spike: { eta: 168, beta: 3.0 },
  normal: { eta: 2160, beta: 1.4 },
  // regímenes extendidos (roadmap #6)
  "casing-leak": { eta: 600, beta: 2.0 },
  "tubing-leak": { eta: 240, beta: 2.8 },
  hydrates: { eta: 300, beta: 2.4 },
  sanding: { eta: 400, beta: 2.5 },
};

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Prior por id de hipótesis (acepta prefijos sensor-pt, spike-pl, …). */
export function priorFor(diagId: string): WeibullPrior {
  if (MODE_PRIORS[diagId]) return MODE_PRIORS[diagId];
  const head = diagId.split("-")[0];
  return MODE_PRIORS[head] ?? MODE_PRIORS.normal;
}

export interface RulCovars {
  anomScore: number; // 0..100
  diagConf: number; // 0..1
  qDropFrac: number; // caída relativa de caudal vs base (0..0.6)
}

const W = { anom: 0.9, conf: 0.7, qdrop: 0.5 }; // pesos AFT

/** η efectiva: la severidad operativa acelera la falla. */
export function etaEffective(eta0: number, c: RulCovars): number {
  const x1 = clamp(c.anomScore, 0, 100) / 100;
  const x2 = clamp(c.diagConf, 0, 1);
  const x3 = clamp(c.qDropFrac, 0, 0.6) / 0.6;
  return eta0 * Math.exp(-(W.anom * x1 + W.conf * x2 + W.qdrop * x3));
}

/** S(t) — probabilidad de seguir sin intervención a t horas. */
export function survivalAt(tHours: number, eta: number, beta: number): number {
  if (tHours <= 0) return 1;
  return Math.exp(-Math.pow(tHours / eta, beta));
}

/** Cuantil incondicional p (horas). */
export function weibullQuantile(p: number, eta: number, beta: number): number {
  return eta * Math.pow(-Math.log(1 - clamp(p, 0, 1 - 1e-9)), 1 / beta);
}

/** Tasa de fallo instantánea a t horas (por hora). */
export function hazardAt(tHours: number, eta: number, beta: number): number {
  if (tHours <= 0) return beta === 1 ? 1 / eta : 0;
  return ((beta / eta) * Math.pow(tHours / eta, beta - 1));
}

/** Cuantil condicional: horas restantes a nivel p dado sobrevivir a tHours. */
export function rulQuantile(tHours: number, p: number, eta: number, beta: number): number {
  const t = Math.max(0, tHours);
  const inner = Math.pow(t / eta, beta) - Math.log(1 - clamp(p, 0, 1 - 1e-9));
  return Math.max(0, eta * Math.pow(inner, 1 / beta) - t);
}

// ----------------------------- edad del episodio ----------------------------

/** Minutos hacia atrás en que la condición se cumple de forma continua. */
export function episodeAgeMin(samples: Sample[], key: VarKey, test: (v: number) => boolean): number {
  if (samples.length === 0) return 0;
  const lastIndex = samples.length - 1;
  if (!test(samples[lastIndex][key])) return 0;
  const lastM = samples[lastIndex].m;
  let firstM = lastM;
  for (let i = lastIndex; i >= 0; i--) {
    if (!test(samples[i][key])) break;
    if (i < lastIndex && samples[i + 1].m - samples[i].m !== 1) break;
    firstM = samples[i].m;
  }
  // No atribuir duración a minutos sin telemetría entre dos muestras.
  return Math.max(1, lastM - firstM + 1);
}

/** Minutos en plano (sensor congelado): tolerancia absoluta 0.2 (como N3). */
export function flatAgeMin(samples: Sample[], key: VarKey): number {
  const s = samples;
  if (s.length === 0) return 0;
  const last = s[s.length - 1][key];
  return episodeAgeMin(s, key, (v) => Math.abs(v - last) < 0.2);
}

const PREDICATES: Partial<
  Record<string, { key: VarKey; test: (base: Record<VarKey, number>) => (v: number) => boolean }>
> = {
  "liquid-loading": { key: "pt", test: (b) => (v) => v < b.pt * 0.96 },
  restriction: { key: "pl", test: (b) => (v) => v < b.pl * 0.97 },
  "control-issue": { key: "choke", test: (b) => (v) => Math.abs(v - b.choke) > 4 },
  sensor: { key: "pt", test: (b) => (v) => Math.abs(v - b.pt) > b.pt * 0.01 },
  spike: { key: "pt", test: (b) => (v) => Math.abs(v - b.pt) > b.pt * 0.01 },
  "casing-leak": { key: "pc", test: (b) => (v) => v < b.pc * 0.985 },
  "tubing-leak": { key: "pt", test: (b) => (v) => v < b.pt * 0.97 },
  hydrates: { key: "temp", test: (b) => (v) => v < b.temp - 1.2 },
  sanding: { key: "q", test: (b) => (v) => Math.abs(v - b.q) > b.q * 0.03 },
};

export interface RulResult {
  mode: string; // id del diagnóstico
  eta0: number; // prior (h)
  beta: number;
  etaEff: number; // vida efectiva con covariables (h)
  ageH: number; // edad estimada del episodio (h)
  rulMedian: number; // horas restantes (cuantil condicional 0.5)
  p10: number;
  p90: number;
  survNow: number; // S(edad)
  hazardNow: number; // por hora
  covars: RulCovars;
}

/**
 * RUL a partir del diagnóstico activo, el buffer y la línea base del pozo.
 * @param diagId id de la hipótesis top (o "normal")
 * @param diagConf confianza del diagnóstico (0..1)
 * @param samples buffer reciente (minuto a minuto)
 * @param base línea base del pozo
 * @param anomScore índice de anomalía actual 0..100
 */
export function rulForDiag(
  diagId: string,
  diagConf: number,
  samples: Sample[],
  base: Record<VarKey, number>,
  anomScore: number,
): RulResult {
  const prior = priorFor(diagId);
  const tailQ = samples.slice(-20).map((s) => s.q);
  const qMean = tailQ.length ? tailQ.reduce((a, b) => a + b, 0) / tailQ.length : base.q;
  const qDropFrac = clamp((base.q - qMean) / Math.max(1, base.q), 0, 0.6);
  // la confianza del diagnóstico "normal" es confianza de que está bien:
  // no debe acelerar la falla (solo la severidad observada cuenta)
  const confEff = diagId === "normal" ? 0 : clamp(diagConf, 0, 1);
  const covars: RulCovars = { anomScore, diagConf: confEff, qDropFrac };
  const etaEff = etaEffective(prior.eta, covars);

  // edad del episodio por modo (min → h)
  let ageMin = 0;
  if (diagId !== "normal") {
    if (diagId.startsWith("sensor-") || diagId.startsWith("spike-")) {
      const key = (diagId.split("-")[1] ?? "pt") as VarKey;
      ageMin = diagId.startsWith("spike-") ? 15 : flatAgeMin(samples, key);
    } else {
      const p = PREDICATES[diagId];
      if (p) ageMin = episodeAgeMin(samples, p.key, p.test(base));
    }
  }
  const ageH = Math.max(0, ageMin / 60);

  return {
    mode: diagId,
    eta0: prior.eta,
    beta: prior.beta,
    etaEff,
    ageH,
    rulMedian: rulQuantile(ageH, 0.5, etaEff, prior.beta),
    p10: rulQuantile(ageH, 0.1, etaEff, prior.beta),
    p90: rulQuantile(ageH, 0.9, etaEff, prior.beta),
    survNow: survivalAt(ageH, etaEff, prior.beta),
    hazardNow: hazardAt(ageH, etaEff, prior.beta),
    covars,
  };
}
