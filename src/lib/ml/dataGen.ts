// ---------------------------------------------------------------------------
// VIGÍA ML · generación de datasets de entrenamiento a partir del simulador
// físico de pozos. Cada pozo sintético corre 520 min con un régimen inyectado;
// de su buffer se extraen ventanas para los tres modelos (N1 / N2 / N3).
// La generación es ASÍNCRONA y cede el hilo principal entre pozos para no
// congelar la interfaz durante la carga.
// ---------------------------------------------------------------------------
import { WellSim } from "../sim";
import type { Sample, Scenario, VarKey, WellCfg } from "../sim";
import { linSlopePerHour } from "../models";

export const ML_VARS: VarKey[] = ["pt", "pc", "pl", "temp", "q", "choke"];

// ------------------------- hiperparámetros de ventanas ---------------------
export const FC_INPUT = 16; // pasos de 15 min a la entrada del LSTM (4 h)
export const FC_STEPS = 6; // pasos previstos por rollout (1.5 h)
export const FC_MIN = 15; // resolución del pronóstico (min)
export const AE_WIN = 30; // ventana del autoencoder (min)
export const CLS_WIN = 90; // ventana de features del clasificador (min)
export const N_FEATURES = 20;

export type ClassId = "normal" | "liquidLoading" | "restriction" | "sensorFault" | "controlIssue";
export const CLASSES: ClassId[] = ["normal", "liquidLoading", "restriction", "sensorFault", "controlIssue"];
export const CLASS_SHORT: Record<ClassId, string> = {
  normal: "Normal",
  liquidLoading: "Liq. loading",
  restriction: "Restricción",
  sensorFault: "Falla sensor",
  controlIssue: "Ctrl. choke",
};
export const CLASS_ABBR: Record<ClassId, string> = {
  normal: "NOR",
  liquidLoading: "LL",
  restriction: "RS",
  sensorFault: "SF",
  controlIssue: "CI",
};

// ------------------------------- utilidades --------------------------------
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const std = (a: number[]) => {
  if (a.length < 2) return 0;
  const m = mean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length);
};

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// escala relativa de normalización por variable (fracción del valor base)
const REL: Record<VarKey, number> = { pt: 0.11, pc: 0.09, pl: 0.12, temp: 0.14, q: 0.09, choke: 0.16 };

export interface Norm {
  base: Record<VarKey, number>;
  scale: Record<VarKey, number>;
}

export function makeNorm(base: Record<VarKey, number>): Norm {
  const scale = {} as Record<VarKey, number>;
  for (const k of ML_VARS) scale[k] = Math.max(0.5, base[k] * REL[k]);
  return { base, scale };
}

export function normVec(s: Sample, n: Norm): number[] {
  return ML_VARS.map((k) => (s[k] - n.base[k]) / n.scale[k]);
}

// --------------------- features para el clasificador (N3) ------------------
// 20 rasgos físicos por ventana: tendencias, diferenciales, oscilaciones y
// DESFASES DE NIVEL respecto a la línea base (clave para detectar regímenes
// ya saturados, donde las pendientes vuelven a cero). Todos acotados a
// [-4, 4] para estabilizar la red densa.
export function extractFeatures(win: Sample[], norm: Norm): number[] {
  const slope = (key: VarKey) => linSlopePerHour(win.map((s) => ({ x: s.m, y: s[key] })));

  const ptT = slope("pt") / 15;
  const pcT = slope("pc") / 15;
  const plT = slope("pl") / 15;
  const qT = slope("q") / Math.max(20, norm.base.q * 0.08);
  const tT = slope("temp") / 3;
  const dpCT = (pcT * 15 - ptT * 15) / 12; // divergencia casing − tubing
  const dpTL = (ptT * 15 - plT * 15) / 12; // diferencial tubing − línea

  const tail30 = win.slice(-30);
  const qTail = tail30.map((s) => s.q);
  const qm = mean(qTail);
  const qOsc = qm > 1 ? std(qTail) / qm / 0.04 : 0; // oscilación tipo slug

  const tail45 = win.slice(-45);
  const chokeRange =
    (Math.max(...tail45.map((s) => s.choke)) - Math.min(...tail45.map((s) => s.choke))) / 10;
  const q45 = tail45.map((s) => s.q);
  const q45m = mean(q45);
  const qResp = q45m > 1 ? std(q45) / q45m / 0.03 : 0; // respuesta de caudal al choke

  const tail25pt = win.slice(-25).map((s) => s.pt);
  const s25 = std(tail25pt);
  const ptFlat = 0.5 / (s25 + 0.15); // alto cuando la señal está congelada
  const m25 = mean(tail25pt);
  const ptZ = Math.max(...tail25pt.map((v) => Math.abs(v - m25) / (s25 || 1e-6))) / 6; // spikes

  const qFirst = mean(win.slice(0, 30).map((s) => s.q));
  const qDrop = qFirst > 1 ? mean(qTail) / qFirst - 1 : 0;
  const pcFirst = mean(win.slice(0, 30).map((s) => s.pc));
  const pcRise = pcFirst > 1 ? mean(tail30.map((s) => s.pc)) / pcFirst - 1 : 0;

  // desfaces de nivel vs línea base del pozo (detectan regímenes saturados)
  const off = (k: VarKey) => (mean(tail30.map((s) => s[k])) - norm.base[k]) / norm.scale[k];
  // apertura del diferencial casing−tubing: firma del liquid loading establecido
  const gapOff =
    ((mean(tail30.map((s) => s.pc)) - mean(tail30.map((s) => s.pt))) -
      (norm.base.pc - norm.base.pt)) / norm.scale.pt;
  // oscilación de P tubing: el slug del LL saturado la eleva (~±9 psi)
  const ptOsc = std(tail45.map((s) => s.pt)) / 9;

  return [ptT, pcT, plT, qT, tT, dpCT, dpTL, qOsc, chokeRange, qResp, ptFlat, ptZ, qDrop, pcRise,
    off("pt"), off("pc"), off("pl"), off("q"), gapOff, ptOsc].map(
    (v) => clamp(v, -4, 4),
  );
}

// --------------------------- dataset completo -------------------------------
export interface Dataset {
  // pronóstico: agregados de 15 min, entrada [16][6] → salida [36]
  fcTrainX: number[][][];
  fcTrainY: number[][];
  fcValX: number[][][];
  fcValY: number[][];
  // autoencoder: ventanas 1-min de operación NORMAL [30][6]
  aeTrainX: number[][][];
  aeValX: number[][][];
  // clasificador: features [20] + etiqueta de clase
  clsTrainX: number[][];
  clsTrainY: number[];
  clsValX: number[][];
  clsValY: number[];
  wells: number;
}

function randomBase(r: () => number): Record<VarKey, number> {
  const u = (a: number, b: number) => a + r() * (b - a);
  return {
    pt: u(470, 680),
    pc: u(660, 900),
    pl: u(380, 480),
    temp: u(36, 48),
    q: u(1500, 3400),
    choke: u(42, 68),
  };
}

/** Convierte el buffer de un pozo en agregados de 15 min normalizados. */
export function aggregates(buf: Sample[], norm: Norm): number[][] {
  const out: number[][] = [];
  for (let i = 0; i + FC_MIN <= buf.length; i += FC_MIN) {
    const agg: number[] = [];
    for (const k of ML_VARS) {
      let s = 0;
      for (let j = 0; j < FC_MIN; j++) s += buf[i + j][k];
      agg.push((s / FC_MIN - norm.base[k]) / norm.scale[k]);
    }
    out.push(agg);
  }
  return out;
}

export interface DatasetOpts {
  /** semilla determinista del generador */
  seed?: number;
  /** 1 = dataset completo (52 pozos) · 0.5 = modo ligero para backend CPU */
  wellsScale?: number;
  /** progreso del generador (0..1); puede devolver Promise para ceder el hilo */
  onProgress?: (frac: number) => void | Promise<void>;
}

export async function generateDataset(opts: DatasetOpts = {}): Promise<Dataset> {
  let s = (opts.seed ?? 20260828) >>> 0;
  const nextSeed = () => ((s = (s * 1664525 + 1013904223) >>> 0), s);

  const mk = (cls: ClassId, i: number): { cfg: WellCfg; cls: ClassId } => {
    const r = mulberry32(nextSeed());
    const base = randomBase(r);
    const at = 60 + Math.floor(r() * 60); // inicio del régimen: deja ventanas saturadas en el buffer
    return {
      cfg: {
        id: `TRN-${CLASS_ABBR[cls]}${i}`,
        name: "pozo sintético de entrenamiento",
        field: "sintético",
        depth: "—",
        base,
        seed: 70000 + i * 137 + Math.floor(r() * 900),
        script: cls === "normal" ? [] : [{ at, s: cls as Scenario }],
        preRun: 520,
      },
      cls,
    };
  };

  const scale = clamp(opts.wellsScale ?? 1, 0.35, 1);
  const nNormal = Math.max(6, Math.round(12 * scale));
  const nFault = Math.max(4, Math.round(10 * scale));

  const plan: { cfg: WellCfg; cls: ClassId }[] = [];
  for (let i = 0; i < nNormal; i++) plan.push(mk("normal", i));
  for (const c of ["liquidLoading", "restriction", "sensorFault", "controlIssue"] as ClassId[]) {
    for (let i = 0; i < nFault; i++) plan.push(mk(c, i));
  }

  const fcTrainX: number[][][] = [], fcTrainY: number[][] = [];
  const fcValX: number[][][] = [], fcValY: number[][] = [];
  const aeTrainX: number[][][] = [], aeValX: number[][][] = [];
  const clsTrainX: number[][] = [], clsTrainY: number[] = [];
  const clsValX: number[][] = [], clsValY: number[] = [];

  const shuffleRnd = mulberry32(nextSeed());
  const shuffledIdx = (len: number) => {
    const a = Array.from({ length: len }, (_, i) => i);
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(shuffleRnd() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  };

  for (let wi = 0; wi < plan.length; wi++) {
    const { cfg, cls } = plan[wi];
    const sim = new WellSim(cfg);
    const norm = makeNorm(cfg.base);
    const at = cfg.script[0]?.at ?? 1e9;
    const buf = sim.buf;
    const isVal = wi % 5 === 0; // split por pozo (sin fuga de datos)

    // --- N1 · ventanas de pronóstico sobre agregados ---
    const aggs = aggregates(buf, norm);
    for (let i = 0; i + FC_INPUT + FC_STEPS <= aggs.length; i++) {
      const x = aggs.slice(i, i + FC_INPUT);
      const y = aggs.slice(i + FC_INPUT, i + FC_INPUT + FC_STEPS).flat();
      if (isVal) { fcValX.push(x); fcValY.push(y); }
      else { fcTrainX.push(x); fcTrainY.push(y); }
    }

    // --- N2 · ventanas normales para el autoencoder ---
    const normalBuf = cls === "normal" ? buf : buf.slice(0, Math.max(0, at - 15));
    const stride = cls === "normal" ? 9 : 12;
    for (let i = 0; i + AE_WIN <= normalBuf.length; i += stride) {
      const w = normalBuf.slice(i, i + AE_WIN).map((sm) => normVec(sm, norm));
      if (isVal) aeValX.push(w);
      else aeTrainX.push(w);
    }

    // --- N3 · ventanas etiquetadas para el clasificador ---
    // Las ventanas de régimen SATURADO (tendencias ya aplanadas, t≥380) se
    // duplican: sin eso el clasificador aprende solo la fase de desarrollo
    // y confunde un régimen establecido con operación normal.
    for (let i = 0; i + CLS_WIN <= buf.length; i += 18) {
      const end = i + CLS_WIN;
      let label: ClassId | null = null;
      if (cls === "normal") label = "normal";
      else if (end <= at - 10) label = "normal"; // fase pre-falla: operación normal
      else if (i >= at + 20) label = cls; // régimen desarrollado
      if (!label) continue; // zona de transición: se descarta
      const f = extractFeatures(buf.slice(i, end), norm);
      const reps = label !== "normal" && end - at >= 380 ? 2 : 1; // oversampling saturado
      for (let k = 0; k < reps; k++) {
        if (isVal) { clsValX.push(f); clsValY.push(CLASSES.indexOf(label)); }
        else { clsTrainX.push(f); clsTrainY.push(CLASSES.indexOf(label)); }
      }
    }

    // ceder el hilo cada 4 pozos: la interfaz respira durante la generación
    if (wi % 4 === 3 || wi === plan.length - 1) {
      await opts.onProgress?.((wi + 1) / plan.length);
    }
  }

  // balanceo: recortar la clase normal si domina demasiado el set de train
  const faultCounts = CLASSES.slice(1).map((c) => clsTrainY.filter((y) => y === CLASSES.indexOf(c)).length);
  const capNormal = 1.4 * Math.max(0, ...faultCounts);
  if (capNormal > 0) {
    let seen = 0;
    const keepX: number[][] = [], keepY: number[] = [];
    for (let i = 0; i < clsTrainY.length; i++) {
      const isNormal = clsTrainY[i] === 0;
      if (isNormal) seen++;
      if (!isNormal || seen <= capNormal) { keepX.push(clsTrainX[i]); keepY.push(clsTrainY[i]); }
    }
    clsTrainX.length = 0; clsTrainX.push(...keepX);
    clsTrainY.length = 0; clsTrainY.push(...keepY);
  }

  // barajar el entrenamiento (orden determinista)
  if (fcTrainX.length > 4) {
    const idx = shuffledIdx(fcTrainX.length);
    const x = idx.map((i) => fcTrainX[i]), y = idx.map((i) => fcTrainY[i]);
    fcTrainX.length = 0; fcTrainX.push(...x);
    fcTrainY.length = 0; fcTrainY.push(...y);
  }

  return {
    fcTrainX, fcTrainY, fcValX, fcValY,
    aeTrainX, aeValX,
    clsTrainX, clsTrainY, clsValX, clsValY,
    wells: plan.length,
  };
}
