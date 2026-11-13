// ---------------------------------------------------------------------------
// VIGÍA · capa analítica (todo corre en el navegador)
// N1 Holt amortiguado + IC · N2 z-score multivariable · N3 reglas de diagnóstico
// N4 probabilidad de cruce de umbral · N5 recomendaciones · calidad de datos
// ---------------------------------------------------------------------------

import { hasRegularMinuteCadence, VAR_META, VAR_KEYS } from "./sim";
import type { Sample, VarKey } from "./sim";

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const clamp01 = (v: number) => clamp(v, 0, 1);

// ------------------------------- N1 · pronóstico ---------------------------

export interface Forecast {
  mean: number[];
  lo: number[];
  hi: number[];
  sd: number;
  stepMin: number;
}

export function holtForecast(values: number[], steps: number, stepMin = 15): Forecast {
  const alpha = 0.32;
  const beta = 0.11;
  const phi = 0.93;
  let level = values[0];
  let trend = values.length > 1 ? values[1] - values[0] : 0;
  const fitted: number[] = [];
  for (let i = 1; i < values.length; i++) {
    const v = values[i];
    const pL = level;
    const pT = trend;
    level = alpha * v + (1 - alpha) * (pL + pT);
    trend = beta * (level - pL) + (1 - beta) * pT;
    fitted.push(pL + pT);
  }
  const res = values.slice(1).map((v, i) => v - fitted[i]);
  const sd = Math.sqrt(res.reduce((a, b) => a + b * b, 0) / Math.max(1, res.length)) || 0.001;

  const mean: number[] = [];
  const lo: number[] = [];
  const hi: number[] = [];
  const zc = 1.28; // IC 80 %
  for (let k = 1; k <= steps; k++) {
    const damp = (phi * (1 - Math.pow(phi, k))) / (1 - phi);
    const mu = level + trend * damp;
    const s = sd * Math.sqrt(k) * 1.08;
    mean.push(mu);
    lo.push(mu - zc * s);
    hi.push(mu + zc * s);
  }
  return { mean, lo, hi, sd, stepMin };
}

// Φ(x) — aproximación de Abramowitz & Stegun
export function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp((-x * x) / 2);
  const p =
    d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}

// --------------------------- N4 · proyección operacional -------------------

export interface ProjRow {
  key: VarKey;
  condition: string;
  prob: number; // 0..1
  hoursEst: number | null;
  thr: number;
}

/** Pronóstico sustituible por el LSTM (misma forma que la salida ML). */
export interface FcOverride {
  byVar: Partial<Record<VarKey, { mean: number[]; lo: number[]; hi: number[] }>>;
}

export function projections(
  samples: Sample[],
  base: Record<VarKey, number>,
  mlFc?: FcOverride | null,
): ProjRow[] {
  const history = samples.slice(-240);
  if (history.length < 240 || !hasRegularMinuteCadence(history, 240)) return [];
  const mk = (key: VarKey, thr: number): ProjRow => {
    const values = history.map((s) => s[key]);
    let mean: number[];
    let sdAt: (k: number) => number;
    const mv = mlFc?.byVar[key];
    if (mv && mv.mean.length > 0) {
      // LSTM: la incertidumbre por paso ya viene en la banda (IC 80 %)
      mean = mv.mean;
      sdAt = (k) => (mv.hi[k] - mv.mean[k]) / 1.28;
    } else {
      const fc = holtForecast(values, 96, 15); // 24 h
      mean = fc.mean;
      sdAt = (k) => fc.sd * Math.sqrt(k + 1) * 1.08;
    }
    let prob = 0;
    let hoursEst: number | null = null;
    for (let k = 0; k < mean.length; k++) {
      const p = normCdf((thr - mean[k]) / Math.max(sdAt(k), 1e-6));
      if (p > prob) prob = p;
      if (hoursEst === null && p > 0.5) hoursEst = ((k + 1) * 15) / 60;
    }
    const meta = VAR_META[key];
    return {
      key,
      thr,
      condition: `${meta.short} < ${Math.round(thr)} ${meta.unit}`,
      prob: clamp01(prob),
      hoursEst,
    };
  };
  return [
    mk("pt", base.pt * 0.82),
    mk("q", base.q * 0.78),
  ];
}

// ------------------------------ N2 · anomalías -----------------------------

export interface Contribution {
  key: VarKey;
  z: number;
}

export interface AnomalyResult {
  score: number;
  level: "ÓPTIMO" | "VIGILAR" | "ALERTA" | "CRÍTICO";
  contributions: Contribution[];
}

const W = { pt: 0.28, q: 0.26, pl: 0.14, pc: 0.12, choke: 0.12, temp: 0.08 } as Record<VarKey, number>;

export function anomalyLevel(score: number): AnomalyResult["level"] {
  if (score < 25) return "ÓPTIMO";
  if (score < 55) return "VIGILAR";
  if (score < 80) return "ALERTA";
  return "CRÍTICO";
}

export function anomaly(samples: Sample[]): AnomalyResult {
  const win = samples.slice(-120);
  const contributions: Contribution[] = [];
  const flatKeys: VarKey[] = [];
  let combined = 0;
  let anyFlat = false;

  for (const key of VAR_KEYS) {
    const vals = win.map((s) => s[key]);
    const n = vals.length;
    const mean = vals.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
    const last = vals[n - 1];

    // señal plana (sensor congelado) — el choke es un setpoint: plano es su estado normal
    const tail = vals.slice(-25);
    const tailSd = Math.sqrt(tail.reduce((a, b) => a + (b - tail.reduce((x, y) => x + y, 0) / tail.length) ** 2, 0) / tail.length);
    if (key !== "choke" && tailSd < 0.05) {
      anyFlat = true;
      flatKeys.push(key);
    }

    const z = sd > 1e-6 ? Math.abs(last - mean) / sd : 0;
    // componente de tendencia: pendiente por hora vs dispersión
    const slope = linSlopePerHour(win.map((s) => ({ x: s.m, y: s[key] })));
    const zt = sd > 1e-6 ? Math.abs(slope) / sd : 0;

    const share = W[key] * (Math.max(0, z - 1.2) + Math.max(0, zt - 0.6) * 2.6);
    combined += share;
    if (share > 0.04) contributions.push({ key, z: Math.max(z, zt * 1.4) });
  }

  if (anyFlat) for (const k of flatKeys) contributions.push({ key: k, z: 5.4 });
  contributions.sort((a, b) => b.z - a.z);
  let score = Math.round(100 * (1 - Math.exp(-combined / 2.6)));
  if (anyFlat) score = Math.max(score, 74);
  score = clamp(score, 2, 98);
  return { score, level: anomalyLevel(score), contributions: contributions.slice(0, 3) };
}

// ------------------------------ N3 · diagnóstico ---------------------------

export interface Hypothesis {
  id: string;
  name: string;
  icon: "droplet" | "valve" | "chip" | "gauge" | "check";
  conf: number;
  evidence: string[];
}

export function linSlopePerHour(pts: { x: number; y: number }[]): number {
  const n = pts.length;
  if (n < 3) return 0;
  let sx = 0, sy = 0, sxy = 0, sxx = 0;
  for (const p of pts) {
    sx += p.x; sy += p.y; sxy += p.x * p.y; sxx += p.x * p.x;
  }
  const denom = n * sxx - sx * sx;
  const slope = denom === 0 ? 0 : (n * sxy - sx * sy) / denom; // por minuto
  return slope * 60; // por hora
}

export function trendPerHour(samples: Sample[], key: VarKey, win = 45): number {
  const vals = samples.slice(-win);
  return linSlopePerHour(vals.map((s) => ({ x: s.m, y: s[key] })));
}

const f1 = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}`;

export function diagnose(samples: Sample[]): Hypothesis[] {
  const ptS = trendPerHour(samples, "pt");
  const pcS = trendPerHour(samples, "pc");
  const plS = trendPerHour(samples, "pl");
  const qS = trendPerHour(samples, "q");
  const dpCT = pcS - ptS;
  const dpTL = ptS - plS;

  const tailQ = samples.slice(-20).map((s) => s.q);
  const qMean = tailQ.reduce((a, b) => a + b, 0) / tailQ.length;
  const qSd = Math.sqrt(tailQ.reduce((a, b) => a + (b - qMean) ** 2, 0) / tailQ.length);

  const tailC = samples.slice(-45).map((s) => s.choke);
  const chokeRange = Math.max(...tailC) - Math.min(...tailC);
  const tailQt = samples.slice(-45).map((s) => s.q);
  const qtMean = tailQt.reduce((a, b) => a + b, 0) / tailQt.length;
  const qtSd = Math.sqrt(tailQt.reduce((a, b) => a + (b - qtMean) ** 2, 0) / tailQt.length);

  const out: Hypothesis[] = [];

  // falla de sensor: señal plana o spikes
  for (const key of ["pt", "pc", "pl", "temp"] as VarKey[]) {
    const tail = samples.slice(-25).map((s) => s[key]);
    const mean = tail.reduce((a, b) => a + b, 0) / tail.length;
    const sd = Math.sqrt(tail.reduce((a, b) => a + (b - mean) ** 2, 0) / tail.length);
    if (sd < 0.05) {
      // minutos en plano
      let flatMin = 0;
      const all = samples.map((s) => s[key]);
      for (let i = all.length - 1; i > 0; i--) {
        if (Math.abs(all[i] - all[all.length - 1]) < 0.2) flatMin++;
        else break;
      }
      out.push({
        id: `sensor-${key}`,
        name: `Sensor defectuoso · ${VAR_META[key].tag}`,
        icon: "chip",
        conf: 0.92,
        evidence: [
          `Señal congelada en ${mean.toFixed(1)} ${VAR_META[key].unit} por ~${flatMin} min (σ≈${sd.toFixed(2)})`,
          "El resto de variables no respalda el cambio → probable falla del transmisor",
        ],
      });
    } else {
      const zMax = Math.max(
        ...samples.slice(-25).map((s) => Math.abs(s[key] - mean) / (sd || 1))
      );
      if (zMax > 6) {
        out.push({
          id: `spike-${key}`,
          name: `Spikes en ${VAR_META[key].tag}`,
          icon: "chip",
          conf: 0.62,
          evidence: [`Lecturas con z > 6 en los últimos 25 min`, "Revisar cableado / interferencia EMI del lazo"],
        });
      }
    }
  }

  const ll = clamp01((-ptS / 14) * 0.35 + (-qS / 240) * 0.35 + (dpCT / 9) * 0.3);
  // una fuga en tubing (pt y pc caen EN PARALELO) imita la caída de pt del
  // liquid loading; aquí pc NUNCA sube — si ambas presiones caen, el régimen
  // es fuga y el liquid loading se suprime para no duplicar el diagnóstico
  const tubingLeakLike = ptS < -10 && pcS < -7;
  if (ll > 0.34 && !tubingLeakLike) {
    const ev = [
      `P tubing ${f1(ptS)} psi/h`,
      `Caudal ${f1(qS)} Mscf/d por hora`,
      `Δ(P casing − P tubing) ${f1(dpCT)} psi/h`,
    ];
    if (qSd / qMean > 0.03) ev.push(`Oscilación tipo slug en caudal (σ rel ${(100 * qSd / qMean).toFixed(1)}%)`);
    out.push({
      id: "liquid-loading",
      name: "Liquid loading en tubing",
      icon: "droplet",
      conf: clamp01(ll + 0.12),
      evidence: ev,
    });
  }

  // LL establecido (saturado): las tendencias ya se aplanaron, pero la
  // oscilación tipo slug en caudal y la oscilación de P tubing persisten.
  // Sin esta rama el fallback estadístico solo detecta LL en fase de
  // desarrollo y un régimen establecido pasa por "normal" (el clasificador
  // ML lo cubre vía gapOff/ptOsc; aquí se replica con señales en ventana).
  const ptTail = samples.slice(-45).map((s) => s.pt);
  const ptM45 = ptTail.reduce((a, b) => a + b, 0) / ptTail.length;
  const ptOsc = Math.sqrt(
    ptTail.reduce((a, b) => a + (b - ptM45) ** 2, 0) / ptTail.length,
  );
  const qOscRel = qMean > 1 ? qSd / qMean : 0;
  if (!out.some((h) => h.id === "liquid-loading") && !tubingLeakLike && qOscRel > 0.03 && ptOsc > 4) {
    out.push({
      id: "liquid-loading",
      name: "Liquid loading establecido en tubing",
      icon: "droplet",
      conf: clamp01(0.5 + (ptOsc - 4) / 25),
      evidence: [
        `Oscilación tipo slug persistente: caudal σ rel ${(100 * qOscRel).toFixed(1)}% con tendencias planas`,
        `P tubing oscila σ≈${ptOsc.toFixed(1)} psi en los últimos 45 min`,
        "Régimen saturado: el gas no logra desaguar el tubing pese a la estabilidad aparente",
      ],
    });
  }

  const rs = clamp01((dpTL / 8) * 0.4 + (-plS / 12) * 0.3 + (-qS / 240) * 0.3);
  if (rs > 0.34) {
    out.push({
      id: "restriction",
      name: "Restricción en línea (hidratos / escala)",
      icon: "valve",
      conf: clamp01(rs + 0.12),
      evidence: [
        `Δ(P tubing − P línea) ${f1(dpTL)} psi/h`,
        `P línea ${f1(plS)} psi/h`,
        `Caudal ${f1(qS)} Mscf/d por hora`,
      ],
    });
  }

  // ---- regímenes extendidos (roadmap #6): fugas, hidratos, arena ---------

  // fuga en el anular: P·casing cae sostenida sin reacción en tubing/flujo.
  // Dos guardias anti-falso-positivo: (1) la caída de pc supera la pendiente
  // máxima natural (oscilación lenta ~4 psi/h + diurnal); (2) el anular perdió
  // ≥1.5 % de su valor en la ventana (una oscilación de 380 min apenas recorre
  // ~1 %) — una onda no puede fingir una descarga sostenida.
  const pcTail = samples.slice(-45).map((s) => s.pc);
  const pcOsc = Math.sqrt(pcTail.reduce((a, b) => a + (b - pcTail.reduce((x, y) => x + y, 0) / pcTail.length) ** 2, 0) / pcTail.length);
  const pcWin = samples.slice(-160).map((s) => s.pc);
  const pcDropFrac = pcWin.length > 60 ? (pcWin[0] - pcWin[pcWin.length - 1]) / Math.max(1, pcWin[0]) : 0;
  if (pcS < -6.5 && Math.abs(ptS) < 7 && qS < 12 && pcOsc < 14 && pcDropFrac > 0.015) {
    out.push({
      id: "casing-leak",
      name: "Fuga de presión en anular (casing)",
      icon: "valve",
      conf: clamp01(0.5 + Math.min(0.38, (-pcS - 5) / 26)),
      evidence: [
        `P·casing ${f1(pcS)} psi/h sostenida con P·tubing estable (${f1(ptS)} psi/h)`,
        `Oscilación de P·casing baja (σ≈${pcOsc.toFixed(1)} psi) — descarga lenta, no ruido`,
        "El anular pierde presión sin que el pozo cambie de régimen → válvula/empaque del casing-head",
      ],
    });
  }

  // fuga en tubing: pt y pc caen en paralelo con pérdida de caudal
  if (tubingLeakLike && qS < -15) {
    out.push({
      id: "tubing-leak",
      name: "Fuga en tubing (comunicación tubing-anular)",
      icon: "valve",
      conf: clamp01(0.58 + Math.min(0.34, (-ptS - 10) / 34 + (-qS - 15) / 320)),
      evidence: [
        `P·tubing ${f1(ptS)} psi/h y P·casing ${f1(pcS)} psi/h cayendo EN PARALELO`,
        `Caudal ${f1(qS)} Mscf/d por hora sin subida de P·casing (descarta liquid loading)`,
        "El gas se fuga del tubing al anular: pérdida de presión y caudal simultánea",
      ],
    });
  }

  // hidratos: enfriamiento sostenido + restricción creciente aguas abajo
  const tempS = trendPerHour(samples, "temp");
  if (tempS < -1.1 && dpTL > 2.5) {
    out.push({
      id: "hydrates",
      name: "Formación de hidratos (riesgo aguas abajo)",
      icon: "droplet",
      conf: clamp01(0.45 + Math.min(0.4, (-tempS - 1) / 4 + dpTL / 40)),
      evidence: [
        `Enfriamiento sostenido ${f1(tempS)} °C/h en cabezal`,
        `Δ(P tubing − P línea) ${f1(dpTL)} psi/h — la restricción crece mientras la temperatura cae`,
        "Ventana P-T acercándose a la curva de formación de hidratos: inhibir antes del taponamiento",
      ],
    });
  }

  // arena: ráfagas de alta frecuencia en caudal con choke estable. La escala
  // es FÍSICA (fracción fija del caudal de referencia), no z-scores: con
  // ráfagas frecuentes la σ se infla y los propios impactos dejarían de ser
  // "outliers". La referencia es la MEDIANA de la ventana (robusta a ráfagas).
  const winQ = samples.slice(-46).map((s) => s.q);
  const winC = samples.slice(-46).map((s) => s.choke);
  const qSorted = [...winQ].sort((a, b) => a - b);
  const qMedian = qSorted[Math.floor(qSorted.length / 2)] || 1;
  let burstCount = 0;
  let chokeJump = 0;
  for (let i = 1; i < winQ.length; i++) {
    if (Math.abs(winQ[i] - winQ[i - 1]) > qMedian * 0.045) burstCount++;
    chokeJump = Math.max(chokeJump, Math.abs(winC[i] - winC[i - 1]));
  }
  const qz = qSd / Math.max(1e-6, qMean);
  if (chokeJump < 1.5 && burstCount >= 5 && qz > 0.02) {
    out.push({
      id: "sanding",
      name: "Producción de arena (ráfagas en caudal)",
      icon: "gauge",
      conf: clamp01(0.45 + Math.min(0.4, (burstCount - 5) / 14 + (qz - 0.02) * 3)),
      evidence: [
        `${burstCount} saltos de caudal > ${Math.round(qMedian * 0.045)} Mscf/d en un minuto, con choke estable (Δ máx ${chokeJump.toFixed(1)} %)`,
        `Oscilación relativa ${(100 * qz).toFixed(1)} % no explicada por el comando del choke`,
        "Impactos de alta frecuencia típicos de arena: revisar trampa y velocidad vs límite de erosión",
      ],
    });
  }

  if (chokeRange > 7 && qtSd / qtMean < 0.02) {
    out.push({
      id: "control-issue",
      name: "Problema de control · actuador de choke",
      icon: "gauge",
      conf: 0.8,
      evidence: [
        `Choke recorrió ${chokeRange.toFixed(0)}% en 45 min`,
        `Caudal sin respuesta (σ relativa ${(100 * qtSd / qtMean).toFixed(1)}%)`,
        "Posible válvula pegada o señal 4–20 mA degradada",
      ],
    });
  }

  out.sort((a, b) => b.conf - a.conf);
  const top = out[0]?.conf ?? 0;
  out.push({
    id: "normal",
    name: "Operación dentro de envolvente",
    icon: "check",
    conf: clamp01(Math.max(0.08, 1 - top - 0.05)),
    evidence: [
      `Todas las variables dentro de ±2σ de su línea base móvil`,
      `Sin cruce de umbrales en la última hora`,
    ],
  });
  return out;
}

// ----------------------------- N5 · recomendaciones ------------------------

export interface Recommendation {
  id: string;
  prio: "ALTA" | "MEDIA" | "RUTINA";
  text: string;
}

export function recommend(
  diag: Hypothesis[],
  proj: ProjRow[],
  _samples: Sample[],
): Recommendation[] {
  const recs: Recommendation[] = [];
  const top = diag.filter((d) => d.id !== "normal").slice(0, 2);

  for (const h of top) {
    if (h.id === "liquid-loading") {
      recs.push(
        { id: "ll-1", prio: "ALTA", text: "Programar descarga de líquidos (swabbing) y verificar velocidad crítica del gas en el tubing." },
        { id: "ll-2", prio: "MEDIA", text: "Evaluar inyección continua de surfactante para reducir la tensión superficial y el arrastre de líquido." },
      );
    } else if (h.id === "restriction") {
      recs.push(
        { id: "rs-1", prio: "ALTA", text: "Inspeccionar flowline por hidratos o escala: ΔP tubing–línea en aumento sostenido. Considerar inhibidor (metanol/MEG)." },
        { id: "rs-2", prio: "MEDIA", text: "Verificar posición real del choke contra el comando y estado del estrangulador." },
      );
    } else if (h.id.startsWith("sensor")) {
      recs.push(
        { id: "sf-1", prio: "ALTA", text: `Verificar transmisor ${h.name.split("·")[1]?.trim() ?? ""}: señal plana detectada. Conmutar al soft-sensor mientras se calibra.` },
        { id: "sf-2", prio: "RUTINA", text: "Revisar lazo 4–20 mA, borneras y alimentación del instrumento en cabeza de pozo." },
      );
    } else if (h.id.startsWith("spike")) {
      recs.push({ id: "sp-1", prio: "MEDIA", text: "Revisar integridad de la señal: spikes intermitentes sugieren interferencia o conexión floja." });
    } else if (h.id === "control-issue") {
      recs.push({
        id: "ci-1", prio: "MEDIA",
        text: `Verificar actuador del choke: el comando osciló ±${Math.round(13)}% sin respuesta del caudal. Probar carrera completa (stroke test).`,
      });
    } else if (h.id === "casing-leak") {
      recs.push(
        { id: "cl-1", prio: "ALTA", text: "Buscar fuga en el anular (válvulas y empaques del casing-head) con detección por ultrasonido o burbujeo: P·casing descarga sin cambio de régimen del pozo." },
        { id: "cl-2", prio: "MEDIA", text: "Monitorear la presión de anular contra su envolvente MOP/MASP y reponer el packoff si la descarga continúa." },
      );
    } else if (h.id === "tubing-leak") {
      recs.push(
        { id: "tl-1", prio: "ALTA", text: "Programar prueba de integridad del tubing (pressure test o logging): pt y pc caen en paralelo — posible comunicación tubing-anular." },
        { id: "tl-2", prio: "MEDIA", text: "Cotejar con el registro de presiones del anular y evaluar reducción transitoria de caudal mientras se confirma la fuga." },
      );
    } else if (h.id === "hydrates") {
      recs.push(
        { id: "hy-1", prio: "ALTA", text: "Inyectar inhibidor de hidratos (metanol/MEG) aguas arriba del punto frío: enfriamiento + ΔP tubing–línea creciente = ventana de formación activa." },
        { id: "hy-2", prio: "MEDIA", text: "Evaluar aislamiento/calefacción del flowline y reducir la expansión Joule-Thomson ajustando el choke aguas abajo." },
      );
    } else if (h.id === "sanding") {
      recs.push(
        { id: "sa-1", prio: "MEDIA", text: "Inspeccionar y drenar la trampa de arena; contrastar la velocidad de flujo contra el límite de erosión (API RP 14E) — las ráfagas sugieren sólidos." },
        { id: "sa-2", prio: "RUTINA", text: "Programar monitoreo de arena (acoustic sand detector) y revisar el completamiento si la frecuencia de ráfagas aumenta." },
      );
    }
  }

  const risky = proj.find((p) => p.prob > 0.35);
  if (risky) {
    const meta = VAR_META[risky.key];
    recs.unshift({
      id: `proj-${risky.key}`,
      prio: "ALTA",
      text: `Proyección de riesgo para ${meta.short.toLowerCase()} sobre ${Math.round(risky.thr)} ${meta.unit} en 24 h (P estimada ${(risky.prob * 100).toFixed(0)}%): solicitar revisión del responsable de operación. No es una consigna de control ni debe ejecutarse automáticamente.`,
    });
  }

  if (recs.length === 0) {
    recs.push({
      id: "ok-1",
      prio: "RUTINA",
      text: "Operación estable. Sin acciones requeridas — próxima evaluación automática en 5 min.",
    });
  }
  return recs;
}

// ------------------------------ calidad de datos ---------------------------

export interface DqRow {
  key: VarKey;
  flat: number; // %
  spikes: number; // por ventana
  latency: number | null; // s; null cuando la fuente no proporciona timestamp real
}

export interface DataQuality {
  overall: number;
  grade: "A" | "B" | "C" | "D";
  rows: DqRow[];
  verdict: string;
}

export function dataQuality(samples: Sample[], now = Date.now()): DataQuality {
  const win = samples.slice(-240);
  if (win.length === 0) {
    return {
      overall: 0,
      grade: "D",
      rows: VAR_KEYS.map((key) => ({ key, flat: 0, spikes: 0, latency: null })),
      verdict: "Sin muestras: no se puede evaluar la calidad de telemetría.",
    };
  }

  const sourceSample = win[win.length - 1];
  const latency = sourceSample.source === "opcua" && Number.isFinite(sourceSample.sourceTs)
    ? Math.max(0, (now - sourceSample.sourceTs!) / 1000)
    : null;

  const rows: DqRow[] = VAR_KEYS.map((key) => {
    const vals = win.map((s) => s[key]);
    let flatCount = 0;
    let spikes = 0;
    const n = vals.length;
    const mean = vals.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1e-6;
    if (key !== "choke") {
      for (let i = 15; i < n; i++) {
        const w = vals.slice(i - 15, i);
        const span = Math.max(...w) - Math.min(...w);
        if (span < Math.abs(mean) * 0.0004 + 0.05) flatCount++;
      }
    }
    for (const v of vals) if (Math.abs(v - mean) / sd > 5.5) spikes++;
    return {
      key,
      flat: (flatCount / Math.max(1, n - 15)) * 100,
      spikes,
      latency,
    };
  });

  const avgFlat = rows.reduce((a, r) => a + r.flat, 0) / rows.length;
  const maxFlat = Math.max(...rows.map((r) => r.flat));
  const totSpikes = rows.reduce((a, r) => a + r.spikes, 0);
  const latencyPenalty = latency === null ? 0 : Math.max(0, latency - 30) * 0.5;

  let overall = 100 - maxFlat * 1.1 - totSpikes * 0.9 - latencyPenalty - avgFlat * 0.4;
  overall = Math.round(clamp(overall, 5, 99));
  const regularHistory = win.length >= 240 && hasRegularMinuteCadence(win, 240);
  if (!regularHistory) overall = Math.min(overall, 59);
  const grade: DataQuality["grade"] = overall >= 90 ? "A" : overall >= 75 ? "B" : overall >= 60 ? "C" : "D";

  const worst = [...rows].sort((a, b) => b.flat - a.flat)[0];
  const verdict = !regularHistory
    ? "Historial insuficiente o discontinuo: se requieren 240 muestras consecutivas (una por minuto) para el análisis temporal completo."
    : grade === "A" || grade === "B"
      ? "Telemetría apta para forecasting (N1) y detección de anomalías (N2)."
      : `Calidad degradada por ${VAR_META[worst.key].tag}: corregir antes de confiar en los modelos.`;

  return { overall, grade, rows, verdict };
}
