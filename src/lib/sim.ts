// ---------------------------------------------------------------------------
// VIGÍA · motor de telemetría sintética (demostración)
// Genera series minuto a minuto por pozo con regímenes operativos inyectables.
// ---------------------------------------------------------------------------

export type VarKey = "pt" | "pc" | "pl" | "temp" | "q" | "choke";
export type Scenario =
  | "normal"
  | "liquidLoading"
  | "restriction"
  | "sensorFault"
  | "controlIssue"
  | "casingLeak"
  | "tubingLeak"
  | "hydrates"
  | "sanding";

export type Severity = "info" | "ok" | "warn" | "alarm";

export interface Sample {
  m: number; // minuto transcurrido (simulado o desde la primera muestra externa)
  ts?: number; // timestamp de origen, epoch ms; ausente en el simulador
  source?: "opcua";
  pt: number;
  pc: number;
  pl: number;
  temp: number;
  q: number;
  choke: number;
}

export interface WellEvent {
  id: number;
  m: number;
  wellId: string;
  severity: Severity;
  msg: string;
}

export const VAR_META: Record<
  VarKey,
  { label: string; short: string; unit: string; dec: number; tag: string; color: string; dir: "pos" | "none" }
> = {
  pt:   { label: "Presión tubing",   short: "P·tubing", unit: "psi",    dec: 0, tag: "PT-101", color: "#4bd1b4", dir: "pos" },
  pc:   { label: "Presión casing",   short: "P·casing", unit: "psi",    dec: 0, tag: "PT-102", color: "#7fb4e6", dir: "pos" },
  pl:   { label: "Presión línea",    short: "P·línea",  unit: "psi",    dec: 0, tag: "PT-108", color: "#e5b45c", dir: "pos" },
  temp: { label: "Temp. cabezal",    short: "Temp",     unit: "°C",     dec: 1, tag: "TT-201", color: "#f08a6c", dir: "none" },
  q:    { label: "Caudal gas",       short: "Caudal",   unit: "Mscf/d", dec: 0, tag: "FT-301", color: "#9cd08f", dir: "pos" },
  choke:{ label: "Apertura choke",   short: "Choke",    unit: "%",      dec: 0, tag: "FV-401", color: "#c9b8f0", dir: "none" },
};

export const VAR_KEYS: VarKey[] = ["pt", "pc", "pl", "temp", "q", "choke"];

// ------------------------- guardia de saneamiento --------------------------
// Toda muestra entra al buffer por un único punto (WellSim.step) y pasa por
// aquí: si un cálculo futuro produce NaN/Infinity (p. ej. un divisor inespe-
// rado en un régimen nuevo), el pipeline estadístico y los modelos ML siguen
// recibiendo valores finitos y coherentes con el rango físico del pozo.
const SAN_LO: Record<VarKey, number> = { pt: 0, pc: 0, pl: 0, temp: -20, q: 0, choke: 0 };
const SAN_HI: Record<VarKey, number> = { pt: 15000, pc: 15000, pl: 15000, temp: 200, q: 60000, choke: 100 };

export function sanitizeSample(s: Sample): Sample {
  const out = { ...s };
  for (const k of VAR_KEYS) {
    if (!Number.isFinite(out[k])) out[k] = k === "temp" ? 0 : SAN_LO[k];
    if (out[k] < SAN_LO[k]) out[k] = SAN_LO[k];
    if (out[k] > SAN_HI[k]) out[k] = SAN_HI[k];
  }
  if (!Number.isFinite(out.m)) out.m = 0;
  if (out.ts !== undefined && (!Number.isFinite(out.ts) || out.ts <= 0)) delete out.ts;
  if (out.source !== undefined && out.source !== "opcua") delete out.source;
  return out;
}

/** Comprueba que una ventana tenga una muestra por cada minuto, sin huecos. */
export function hasRegularMinuteCadence(samples: Sample[], required = samples.length): boolean {
  if (!Number.isInteger(required) || required < 1 || samples.length < required) return false;
  const win = samples.slice(-required);
  return win.every((sample, i) => i === 0 || sample.m - win[i - 1].m === 1);
}

// el minuto 0 equivale a ~6.7 h atrás en tiempo real
export const SIM_EPOCH = Date.now() - 400 * 60000;

export const fmtClock = (m: number) =>
  new Date(SIM_EPOCH + m * 60000).toTimeString().slice(0, 5);

export const fmtDayTime = (m: number) => {
  const d = new Date(SIM_EPOCH + m * 60000);
  return `${d.getDate().toString().padStart(2, "0")}/${(d.getMonth() + 1)
    .toString()
    .padStart(2, "0")} ${d.toTimeString().slice(0, 5)}`;
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

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export interface WellCfg {
  id: string;
  name: string;
  field: string;
  depth: string;
  base: Record<VarKey, number>;
  seed: number;
  script: { at: number; s: Scenario }[];
  preRun: number;
}

export const SCENARIO_INFO: Record<Scenario, { label: string; sev: Severity; msg: string }> = {
  normal:       { label: "Normal",          sev: "ok",    msg: "Régimen restablecido a operación normal" },
  liquidLoading:{ label: "Liquid loading",  sev: "warn",  msg: "Régimen inyectado: liquid loading (demo)" },
  restriction:  { label: "Restricción",     sev: "warn",  msg: "Régimen inyectado: restricción en línea (demo)" },
  sensorFault:  { label: "Falla de sensor", sev: "alarm", msg: "Falla de sensor inyectada en PT-101 (demo)" },
  controlIssue: { label: "Problema de control", sev: "warn", msg: "Régimen inyectado: actuador de choke sin respuesta (demo)" },
  casingLeak:   { label: "Fuga en anular",  sev: "warn",  msg: "Régimen inyectado: fuga de presión en anular (demo)" },
  tubingLeak:   { label: "Fuga en tubing",  sev: "alarm", msg: "Régimen inyectado: fuga en tubing (demo)" },
  hydrates:     { label: "Hidratos",        sev: "warn",  msg: "Régimen inyectado: formación de hidratos (demo)" },
  sanding:      { label: "Arena",           sev: "warn",  msg: "Régimen inyectado: impactos de arena (demo)" },
};

let evtSeq = 1;

export class WellSim {
  id: string;
  name: string;
  field: string;
  depth: string;
  base: Record<VarKey, number>;
  buf: Sample[] = [];
  events: WellEvent[] = [];
  m = 0;
  phaseT = 0;
  scenario: Scenario = "normal";
  score = 0;
  level = "ÓPTIMO";

  private rnd: () => number;
  private m0: number;
  private choke: number;
  private chokeTarget: number;
  private lastAdj: number;
  private frozen: number | null = null;
  private script: { at: number; s: Scenario }[];

  constructor(cfg: WellCfg) {
    this.id = cfg.id;
    this.name = cfg.name;
    this.field = cfg.field;
    this.depth = cfg.depth;
    this.base = { ...cfg.base };
    this.script = cfg.script;
    this.rnd = mulberry32(cfg.seed);
    this.m0 = Math.floor(cfg.seed % 900);
    this.choke = cfg.base.choke;
    this.chokeTarget = cfg.base.choke;
    this.lastAdj = 0;
    for (let i = 0; i < cfg.preRun; i++) {
      this.m++;
      for (const sc of cfg.script) {
        if (this.m === sc.at) this.applyScenario(sc.s, true);
      }
      this.step();
    }
    this.log("info", "Consola conectada · suscripción OPC-UA activa (1 min)");
  }

  get last(): Sample {
    return this.buf[this.buf.length - 1];
  }

  tick() {
    this.m++;
    this.step();
  }

  setScenario(s: Scenario) {
    this.applyScenario(s, true);
  }

  log(sev: Severity, msg: string) {
    this.events.push({ id: evtSeq++, m: this.m, wellId: this.id, severity: sev, msg });
    if (this.events.length > 140) this.events.shift();
  }

  private applyScenario(s: Scenario, announce: boolean) {
    this.scenario = s;
    this.phaseT = 0;
    if (s === "sensorFault") this.frozen = this.buf.length ? this.last.pt : this.base.pt;
    if (s === "normal") {
      this.frozen = null;
      this.chokeTarget = this.base.choke;
    }
    if (announce) {
      const info = SCENARIO_INFO[s];
      this.log(info.sev, info.msg);
    }
  }

  private step() {
    const r = this.rnd;
    const b = this.base;
    const t = this.phaseT;
    const m = this.m;
    const n = (a: number) => (r() * 2 - 1) * a;

    const diurnal = Math.sin((((m + this.m0) % 1440) / 1440) * Math.PI * 2);
    const slow = Math.sin(((m + this.m0) / 380) * Math.PI * 2);

    // --- dinámica del choke ---
    if (this.scenario === "controlIssue") {
      this.chokeTarget = b.choke + 13 * Math.sin(t / 22);
    } else if (this.scenario === "normal" && m - this.lastAdj > 60 + r() * 50) {
      this.lastAdj = m;
      const d = (r() < 0.5 ? -1 : 1) * (2 + Math.round(r() * 2));
      const nt = clamp(this.chokeTarget + d, b.choke - 8, b.choke + 8);
      this.log("info", `Ajuste de choke ${Math.round(this.choke)}% → ${Math.round(nt)}% (operador)`);
      this.chokeTarget = nt;
    }
    this.choke += (this.chokeTarget - this.choke) * 0.18;

    // el problema de control desacopla el caudal del choke
    const chokeGain = this.scenario === "controlIssue" ? 0.1 : 1;
    const chokeFactor = Math.pow(this.choke / b.choke, 0.85 * chokeGain);

    let pt = b.pt + diurnal * 3.5 + slow * 5 + n(1.6);
    let pc = b.pc + diurnal * 2.5 + slow * 4 + n(1.4);
    let pl = b.pl + diurnal * 2 + slow * 3 + n(1.1);
    let temp = b.temp + diurnal * 1.2 + n(0.35);
    let q = b.q * chokeFactor * (1 + diurnal * 0.015) + n(b.q * 0.012);

    if (this.scenario === "liquidLoading") {
      const s = Math.min(1, t / 420);
      pt -= 48 * s + Math.sin(t / 6) * 9 * s;
      pc += 30 * s + Math.sin(t / 6 + 1) * 5 * s;
      temp -= 5 * s;
      pl -= 6 * s;
      const slug = s > 0.3 ? Math.sin(t / 4.5) * b.q * 0.045 * s : 0;
      q = q * (1 - 0.26 * s) + slug - (r() < 0.015 * s ? b.q * 0.08 : 0);
    } else if (this.scenario === "restriction") {
      const s = Math.min(1, t / 300);
      pl -= 62 * s;
      pt += 13 * s;
      temp += 2.5 * s;
      q = q * (1 - 0.2 * s);
    } else if (this.scenario === "casingLeak") {
      // fuga en el anular: P·casing cae sostenida, el resto casi no reacciona
      const s = Math.min(1, t / 360);
      pc -= 58 * s;
      q = q * (1 - 0.06 * s);
    } else if (this.scenario === "tubingLeak") {
      // fuga en tubing: pt y pc caen juntas (comunicación), el caudal cae;
      // a diferencia del liquid loading, aquí pc NO sube
      const s = Math.min(1, t / 300);
      pt -= 62 * s;
      pc -= 44 * s;
      temp -= 3 * s;
      q = q * (1 - 0.24 * s);
    } else if (this.scenario === "hydrates") {
      // hidratos: caída térmica + restricción aguas abajo + pérdida de caudal
      const s = Math.min(1, t / 330);
      temp -= 7 * s;
      pl -= 50 * s;
      pt += 11 * s;
      q = q * (1 - 0.18 * s);
    } else if (this.scenario === "sanding") {
      // arena: ráfagas de alta frecuencia en caudal con choke estable + jitter
      // de P·línea (impactos contra la trampa/restricciones locales).
      // Rampa corta (45 min): la arena aparece por cambios de drawdown, no es
      // un proceso lento — el demo la muestra en ~100 s de reloj real.
      const s = Math.min(1, t / 45);
      q = q + Math.sin(t / 1.8) * b.q * 0.022 * s + (r() < 0.1 ? (r() < 0.5 ? -1 : 1) * b.q * 0.07 * s : 0);
      pl += Math.sin(t / 2.4) * 7 * s + n(2 * s);
      pt += Math.sin(t / 3.1) * 5 * s;
    } else if (this.scenario === "sensorFault") {
      pt = (this.frozen ?? b.pt) + n(0.02);
      if (r() < 0.012) pt = (this.frozen ?? b.pt) + 220 * (r() < 0.5 ? -1 : 1);
    }

    q = Math.max(0, q);
    this.buf.push(sanitizeSample({ m, pt, pc, pl, temp, q, choke: clamp(this.choke, 2, 100) }));
    if (this.buf.length > 520) this.buf.shift();
    this.phaseT++;
  }
}

// ---------------------------------------------------------------------------
// flota de demostración
// ---------------------------------------------------------------------------
export function createFleet(): WellSim[] {
  const cfgs: WellCfg[] = [
    {
      id: "PN-041", name: "Piedemonte Norte 41", field: "Piedemonte Llanero", depth: "3.842 m",
      base: { pt: 560, pc: 780, pl: 430, temp: 44, q: 2250, choke: 54 },
      seed: 1103, script: [{ at: 175, s: "liquidLoading" }], preRun: 312,
    },
    {
      id: "PN-017", name: "Piedemonte Norte 17", field: "Piedemonte Llanero", depth: "4.105 m",
      base: { pt: 605, pc: 820, pl: 445, temp: 41, q: 2680, choke: 61 },
      seed: 2287, script: [{ at: 225, s: "restriction" }], preRun: 312,
    },
    {
      id: "AL-006", name: "Altiplano 6", field: "Cordillera Oriental", depth: "2.976 m",
      base: { pt: 500, pc: 705, pl: 402, temp: 38, q: 1830, choke: 47 },
      seed: 3517, script: [{ at: 252, s: "sensorFault" }], preRun: 312,
    },
    {
      id: "GN-118", name: "Gasoducto Sur 118", field: "Cuenca del Sur", depth: "3.310 m",
      base: { pt: 640, pc: 860, pl: 458, temp: 46, q: 3120, choke: 63 },
      seed: 4861, script: [], preRun: 312,
    },
    {
      id: "CS-233", name: "Cuenca Este 233", field: "Cuenca del Este", depth: "2.744 m",
      base: { pt: 585, pc: 798, pl: 440, temp: 43, q: 2410, choke: 58 },
      seed: 6229, script: [], preRun: 312,
    },
  ];
  return cfgs.map((c) => new WellSim(c));
}

export function mergeEvents(wells: WellSim[], limit = 40): WellEvent[] {
  return wells
    .flatMap((w) => w.events)
    .sort((a, b) => b.m - a.m || b.id - a.id)
    .slice(0, limit);
}
