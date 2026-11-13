// ---------------------------------------------------------------------------
// VIGÍA · copiloto «pregúntale al pozo» (roadmap comercial #2)
// Motor de consulta en lenguaje natural 100 % determinista y offline: interpreta
// la pregunta (español con sinónimos e inglés básico), la clasifica por intención
// y compone la respuesta con las salidas reales del pipeline (N1–N5, RUL Weibull,
// DCA Arps, medición virtual, calidad de datos, eventos, flota).
// Sin red, sin API externa, sin modelo generativo: la misma respuesta para la
// misma entrada y cero fuga de datos del navegador — coherente con la promesa
// de privacidad del producto. En producción podría enchufarse un LLM opcional
// detrás de esta misma interfaz (answerQuestion).
// ---------------------------------------------------------------------------

import { VAR_META } from "./sim";
import type { Sample, VarKey, WellEvent } from "./sim";
import type {
  AnomalyResult,
  Contribution,
  DataQuality,
  Hypothesis,
  ProjRow,
  Recommendation,
} from "./models";
import { trendPerHour } from "./models";
import { rulForDiag } from "./rul";
import { dailyHistory } from "./wellHistory";
import { fitArps, monthlyDeclinePct } from "./arps";
import { meterCheck, suggestDia } from "./virtualMeter";
import { adviseSetpoints } from "./setpoints";
import { calibrateTwin, twinGapRecent } from "./twin";
import { explainSamples } from "./explain";
import type { ExplainResult } from "./explain";

export interface FleetSummaryRow {
  id: string;
  name: string;
  level: string;
  score: number;
  qNow: number;
}

/** Todo lo que el copiloto puede contar: el estado vivo del pozo y la flota. */
export interface CopilotContext {
  well: { id: string; name: string; field: string; depth: string };
  samples: Sample[];
  base: Record<VarKey, number>;
  anom: { score: number; level: AnomalyResult["level"]; contributions: Contribution[]; ml: boolean };
  diag: Hypothesis[];
  diagMl: boolean;
  proj: ProjRow[];
  recs: Recommendation[];
  dq: DataQuality;
  events: WellEvent[];
  fleet: FleetSummaryRow[];
}

export interface CopilotAnswer {
  intent: string;
  answer: string; // narrativa principal (admite saltos de línea)
  bullets: string[]; // datos de apoyo
}

export const SUGGESTED_QUESTIONS: string[] = [
  "¿Cómo está el pozo?",
  "¿Qué le pasa a este pozo?",
  "¿Por qué subió el índice de anomalía?",
  "¿Cuándo se estima la falla?",
  "¿Cuál es el choke óptimo?",
  "¿Cuál es el EUR del pozo?",
  "¿El medidor concuerda con la medición virtual?",
  "¿Cómo está el gemelo digital?",
  "¿Cuál es el peor pozo de la flota?",
  "Dame un resumen narrativo",
];

// ------------------------------- utilidades --------------------------------

/** minúsculas sin acentos ni puntuación — hace la consulta insensible a tildes. */
export function normalizeQ(s: string): string {
  return s
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[¿?¡!.,;:()"]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const nf = (v: number, dec = 0) =>
  Number.isFinite(v)
    ? v.toLocaleString("es-EC", { maximumFractionDigits: dec, minimumFractionDigits: dec })
    : "—";

const pct = (v: number) => `${nf(v * 100, 0)} %`;

const SIGN = (v: number) => (v >= 0 ? "+" : "");

/** Horas legibles: "≈ 34 h" / "≈ 3,2 d" cuando supera 72 h. */
export function fmtHours(h: number): string {
  if (!Number.isFinite(h)) return "—";
  if (h >= 72) return `≈ ${nf(h / 24, 1)} d`;
  return `≈ ${nf(h)} h`;
}

const MODE_LABEL: Record<string, string> = {
  "liquid-loading": "carga de líquidos",
  restriction: "restricción en línea",
  "control-issue": "actuador de choke",
  normal: "operación normal",
  "casing-leak": "fuga en anular",
  "tubing-leak": "fuga en tubing",
  hydrates: "formación de hidratos",
  sanding: "producción de arena",
};
const modeLabel = (id: string) =>
  MODE_LABEL[id] ??
  (id.startsWith("sensor-") || id.startsWith("spike-")
    ? `instrumentación (${id.split("-")[1]?.toUpperCase() ?? "?"})`
    : id);

/** Coincidencia por palabra completa (tras normalizar) — evita que "q" case con "que". */
function hasWord(q: string, word: string): boolean {
  const re = new RegExp(`(^| )${word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}( |$)`);
  return re.test(q);
}

// ------------------------- cálculos bajo demanda ---------------------------

/** DCA sobre el histórico diario determinista del pozo (180 d). */
function dcaSummary(ctx: CopilotContext) {
  const hist = dailyHistory(ctx.well.id, ctx.base.q);
  const fit = fitArps(hist.t, hist.q);
  return { hist, fit, monthly: fit.feasible ? monthlyDeclinePct(fit) : null };
}

/** Medición virtual con el mismo criterio de VirtualMeterCard. */
function vmSummary(ctx: CopilotContext) {
  const last = ctx.samples[ctx.samples.length - 1];
  const dia = suggestDia(ctx.base.q, ctx.base.pt, ctx.base.pl, ctx.base.temp, ctx.base.choke);
  const chk = meterCheck(
    {
      chokePct: last.choke,
      pUpPsig: last.pt,
      pDownPsig: last.pl,
      tempC: last.temp,
      chokeDiaIn: dia,
    },
    last.q,
  );
  return { dia, chk };
}

/** RUL del diagnóstico activo (o "normal"). */
function rulSummary(ctx: CopilotContext) {
  const top = ctx.diag.find((d) => d.id !== "normal") ?? ctx.diag[ctx.diag.length - 1] ?? { id: "normal", conf: 0 };
  return rulForDiag(top.id, top.conf, ctx.samples, ctx.base, ctx.anom.score);
}

/** Asesor de setpoints con el mismo criterio de SetpointAdvisorCard. */
function setpointSummary(ctx: CopilotContext) {
  const top = ctx.diag.find((d) => d.id !== "normal") ?? ctx.diag[ctx.diag.length - 1] ?? { id: "normal", conf: 0 };
  return adviseSetpoints({ samples: ctx.samples, base: ctx.base, diagId: top.id, diagConf: top.conf, anomScore: ctx.anom.score });
}

/** Calibración del gemelo sobre la ventana estándar. */
function twinSummary(ctx: CopilotContext) {
  const cal = calibrateTwin(ctx.samples);
  return { cal, gap: cal.feasible ? twinGapRecent(ctx.samples, cal) : null };
}

/** Explicabilidad del índice de anomalía sobre la ventana estándar. */
export function explainOf(ctx: CopilotContext): ExplainResult {
  return explainSamples(ctx.samples, { score: ctx.anom.score, level: ctx.anom.level });
}

// ----------------------------- intent builders -----------------------------

type Builder = (ctx: CopilotContext, q: string) => CopilotAnswer;

const ansEstado: Builder = (ctx) => {
  const last = ctx.samples[ctx.samples.length - 1];
  const qPct = (last.q / Math.max(1, ctx.base.q)) * 100;
  const top = ctx.diag[0];
  const alarms = ctx.fleet.filter((w) => w.level === "ALERTA" || w.level === "CRÍTICO").length;
  return {
    intent: "estado",
    answer:
      `«${ctx.well.id} — ${ctx.well.name}» (${ctx.well.field}) está en nivel ${ctx.anom.level} con índice de anomalía ${ctx.anom.score}/100` +
      `${ctx.anom.ml ? " (autoencoder N2)" : " (z-score multivariable)"}. ` +
      `Caudal ${nf(last.q)} Mscf/d (${nf(qPct)} % de la base ${nf(ctx.base.q)}), P·tubing ${nf(last.pt)} psi, P·casing ${nf(last.pc)} psi, choke ${nf(last.choke)} %.`,
    bullets: [
      `Diagnóstico principal: ${top?.name ?? "—"} (confianza ${pct(top?.conf ?? 0)})${ctx.diagMl ? " · fusionado con clasificador ML" : ""}`,
      `Calidad de telemetría: ${ctx.dq.grade} (${ctx.dq.overall}/100)`,
      `Flota: ${alarms} pozo(s) en ALERTA o CRÍTICO de ${ctx.fleet.length}`,
    ],
  };
};

const ansDiagnostico: Builder = (ctx) => {
  const activos = ctx.diag.filter((d) => d.id !== "normal");
  const top = activos[0] ?? ctx.diag[ctx.diag.length - 1];
  const bullets = [...(top?.evidence ?? [])];
  const second = activos[1];
  if (second) bullets.push(`Hipótesis secundaria: ${second.name} (confianza ${pct(second.conf)})`);
  return {
    intent: "diagnostico",
    answer: activos.length
      ? `Diagnóstico activo: ${top.name} con confianza ${pct(top.conf)}${ctx.diagMl ? " (fusión estadística + clasificador ML N3)" : ""}. ${activos.length > 1 ? `Hay ${activos.length} hipótesis en curso; la de mayor respaldo es la principal.` : "Es la única hipótesis con respaldo en la ventana actual."}`
      : `Sin fallas activas: todas las variables operan dentro de su envolvente (confianza de operación normal ${pct(top?.conf ?? 1)})${ctx.diagMl ? " — el clasificador ML no contradice el diagnóstico" : ""}.`,
    bullets,
  };
};

const ansCaudal: Builder = (ctx) => {
  const last = ctx.samples[ctx.samples.length - 1];
  const trend = trendPerHour(ctx.samples, "q");
  const projQ = ctx.proj.find((p) => p.key === "q");
  const bullets: string[] = [
    `Tendencia reciente: ${SIGN(trend)}${nf(trend, 1)} Mscf/d por hora`,
    `Línea base del pozo: ${nf(ctx.base.q)} Mscf/d · apertura de choke ${nf(last.choke)} %`,
  ];
  if (projQ) {
    bullets.push(
      `Proyección 24 h: ${pct(projQ.prob)} de probabilidad de cruzar ${nf(projQ.thr)} Mscf/d` +
        (projQ.hoursEst !== null ? ` (cruce estimado en ${nf(projQ.hoursEst, 1)} h)` : ""),
    );
  }
  const delta = ((last.q - ctx.base.q) / Math.max(1, ctx.base.q)) * 100;
  const relTxt =
    delta >= 0
      ? `por encima de la base (+${nf(delta, 1)} %)`
      : `${nf(delta, 1)} % por debajo de la base`;
  return {
    intent: "caudal",
    answer:
      `Caudal actual ${nf(last.q)} Mscf/d (${relTxt}), con tendencia ${trend >= 0 ? "al alza" : "a la baja"} de ${SIGN(trend)}${nf(trend, 1)} Mscf/d por hora en la última ventana.`,
    bullets,
  };
};

const ansPresion: Builder = (ctx, q) => {
  const key: VarKey = hasWord(q, "casing") ? "pc" : hasWord(q, "linea") ? "pl" : "pt";
  const meta = VAR_META[key];
  const last = ctx.samples[ctx.samples.length - 1];
  const trend = trendPerHour(ctx.samples, key);
  const baseV = ctx.base[key];
  const delta = ((last[key] - baseV) / Math.max(1, baseV)) * 100;
  const contrib = ctx.anom.contributions.find((c) => c.key === key);
  const projP = ctx.proj.find((p) => p.key === key);
  const bullets: string[] = [
    `Línea base: ${nf(baseV)} psi · tendencia ${SIGN(trend)}${nf(trend, 1)} psi/h`,
  ];
  if (contrib) bullets.push(`Esta variable figura entre los drivers del índice de anomalía (z ${nf(contrib.z, 1)})`);
  if (projP) bullets.push(`Proyección 24 h: ${pct(projP.prob)} de cruzar ${nf(projP.thr)} psi`);
  if (key === "pt") {
    const dpCT = trendPerHour(ctx.samples, "pc") - trend;
    bullets.push(`Δ(P·casing − P·tubing) evolucionando ${SIGN(dpCT)}${nf(dpCT, 1)} psi/h — diferencial clave de liquid loading`);
  }
  return {
    intent: "presion",
    answer: `${meta.label} (${meta.tag}) en ${nf(last[key])} psi, ${delta >= 0 ? "+" : ""}${nf(delta, 1)} % respecto a la base y ${trend >= 0 ? "subiendo" : "cayendo"} ${SIGN(trend)}${nf(trend, 1)} psi/h.`,
    bullets,
  };
};

const ansAnomalia: Builder = (ctx) => {
  const ex = explainOf(ctx);
  const top = ex.drivers.slice(0, 3);
  return {
    intent: "anomalia",
    answer: `${ex.summary}${ctx.anom.ml ? " Score calculado por el autoencoder (N2); la descomposición usa la misma ponderación estadística." : ""}`,
    bullets: top.map((d) => d.text),
  };
};

const ansRul: Builder = (ctx) => {
  const r = rulSummary(ctx);
  const active = r.mode !== "normal";
  const bullets: string[] = [
    `Supervivencia actual S(t): ${nf(r.survNow * 100, 1)} % · edad del episodio ${nf(r.ageH, 1)} h`,
    `η efectiva ${nf(r.etaEff)} h (prior ${nf(r.eta0)} h) · β ${nf(r.beta, 1)} · hazard ${nf(r.hazardNow, 4)}/h`,
    `Covariables de aceleración: anomalía ${ctx.anom.score}/100, caída de caudal ${pct(r.covars.qDropFrac)}`,
  ];
  return {
    intent: "rul",
    answer: active
      ? `Con el modo «${modeLabel(r.mode)}» en curso, la falla se estima en ${fmtHours(r.rulMedian)} (rango p10–p90: ${fmtHours(r.p10)} – ${fmtHours(r.p90)}), condicional a la severidad observada (Weibull + aceleración AFT).`
      : `Sin episodio de degradación activo: no aplica "falla estimada". Si apareciera liquid loading con la severidad actual, la vida mediana previa por prior Weibull ronda ${fmtHours(r.eta0 * Math.pow(Math.LN2, 1 / r.beta))}; el monitor lo recalcularía al instante.`,
    bullets,
  };
};

const ansDeclinacion: Builder = (ctx) => {
  const { fit, monthly } = dcaSummary(ctx);
  if (!fit.feasible) {
    return {
      intent: "declinacion",
      answer: `No fue posible ajustar una declinación defendible sobre el histórico del pozo (${fit.reason ?? "serie no explicada por Arps"}). Se rechaza el ajuste en lugar de inventar un EUR.`,
      bullets: [],
    };
  }
  return {
    intent: "declinacion",
    answer:
      `El histórico de ${fit.n} días ajusta a una declinación ${fit.model} (R² ${nf(fit.r2, 3)}): qi ${nf(fit.qi)} Mscf/d, Di efectiva ${nf(monthly ?? 0, 1)} %/mes, b ${nf(fit.b, 2)}. ` +
      `EUR a abandono (5 % de qi): ${nf(fit.eur / 1000, 1)} MMscf${fit.tAbandonDays !== null ? ` en ${nf(fit.tAbandonDays / 365, 1)} años` : " (excede el horizonte de 30 años)"}; a 5 años acumula ${nf(fit.eur5y / 1000, 1)} MMscf.`,
    bullets: [
      `EUR a 1 año: ${nf(fit.eur1y / 1000, 1)} MMscf · RMSE del ajuste ${nf(fit.rmse, 1)} Mscf/d`,
      `El pronóstico alimenta el reporte PDF y la proyección de reservas por pozo`,
    ],
  };
};

const ansMedidor: Builder = (ctx) => {
  const { dia, chk } = vmSummary(ctx);
  const ok = chk.cdCalibrated !== null && chk.qVirtual > 1;
  return {
    intent: "medidor",
    answer: ok
      ? `Medición virtual (Bean, choke efectivo ${nf(dia, 2)}"): Q virtual ${nf(chk.qVirtual)} Mscf/d vs medidor ${nf(chk.qMeasured)} Mscf/d — divergencia ${SIGN(chk.devPct)}${nf(chk.devPct, 1)} %, ${chk.flag ? "FUERA de tolerancia (> 5 %): posible decaimiento del FT o cambio de régimen no capturado" : "dentro de tolerancia (±5 %)"}. Cd calibrado ${nf(chk.cdCalibrated ?? 0, 2)}.`
      : `No es posible una calibración defendible en las condiciones actuales (rango físico del choke fuera de límites): se abstiene en lugar de inventar un Cd.`,
    bullets: [
      `Fórmula de Bean + factor subcrítico; Cd se calibra contra el medidor fiscal`,
      `Una divergencia sostenida > 5 % dispara la bandera de verificación del medidor`,
    ],
  };
};

const ansSetpoints: Builder = (ctx) => {
  const adv = setpointSummary(ctx);
  if (!adv.feasible) {
    return {
      intent: "setpoints",
      answer: `El asesor de setpoints está en espera: ${adv.reason ?? "condiciones insuficientes"}. Se abstiene en lugar de sugerir una apertura sin base física.`,
      bullets: [],
    };
  }
  const move = adv.chokeRec - adv.chokeNow;
  const accion = Math.abs(move) <= 1.5 ? "mantener la apertura actual" : move > 0 ? `abrir a ${nf(adv.chokeRec)} %` : `cerrar a ${nf(adv.chokeRec)} %`;
  return {
    intent: "setpoints",
    answer:
      `Choke óptimo al horizonte de ${nf(adv.horizonH)} h: ${accion} (actual ${nf(adv.chokeNow)} %). ` +
      `Caudal esperado ${nf(adv.qRec)} Mscf/d (${adv.gainPct >= 0 ? "+" : ""}${nf(adv.gainPct, 1)} %) con supervivencia Weibull ${(adv.survRec * 100).toFixed(1)} % — la utilidad producción×supervivencia es máxima ahí. ` +
      `Volumen adicional esperado: ${nf(adv.expectedDeltaMscf / 1000, 2)} MMscf en ${nf(adv.horizonH)} h.`,
    bullets: [
      ...adv.constraints.map((c) => `${c.ok ? "OK" : "VIOLADO"} · ${c.label}: ${c.detail}`),
      ...adv.rationale.slice(0, 2),
    ],
  };
};

const ansGemelo: Builder = (ctx) => {
  const { cal, gap } = twinSummary(ctx);
  if (!cal.feasible) {
    return {
      intent: "gemelo",
      answer: `El gemelo digital no se calibra con la ventana actual: ${cal.reason ?? "sin excitación"}. Mueve el choke (o espera un ajuste del operador) y reintenta.`,
      bullets: [],
    };
  }
  const g = gap;
  return {
    intent: "gemelo",
    answer:
      `Gemelo calibrado con grado ${cal.grade} (${cal.quality}/100, N=${cal.n} muestras, excitación ${nf(cal.chokeRange, 1)} % de choke): ` +
      `q ∝ choke^${nf(cal.k, 2)}, drawdown a ${nf(cal.a, 3)} psi/Mscf, ganancia de casing b ${nf(cal.b, 3)}. ` +
      `Brecha actual (medido − gemelo) ${g && g.gapPct >= 0 ? "+" : ""}${nf(g?.gapPct ?? NaN, 1)} % — ${g?.misaligned ? "DESALINEADO: disparar recalibración y revisar instrumentos" : "alineado, apto para detección de deriva"}.`,
    bullets: [
      cal.verdict,
      `NRMSE q ${nf(cal.nrmseQ, 3)} · pt ${nf(cal.nrmsePt, 3)} · pc ${nf(cal.nrmsePc, 3)}`,
      "La brecha sostenida > 6 % sugiere deriva de instrumentos o cambio del yacimiento no capturado",
    ],
  };
};

const ansEventos: Builder = (ctx) => {
  const alarmas = ctx.events.filter((e) => e.severity === "alarm").length;
  const avisos = ctx.events.filter((e) => e.severity === "warn").length;
  return {
    intent: "eventos",
    answer: `Registro reciente de la flota: ${ctx.events.length} eventos (${alarmas} alarmas, ${avisos} avisos, ${ctx.events.length - alarmas - avisos} informativos).`,
    bullets: ctx.events.slice(0, 4).map((e) => `${e.severity === "alarm" ? "ALARMA" : e.severity === "warn" ? "aviso" : "info"} · ${e.wellId} · ${e.msg}`),
  };
};

const ansCalidad: Builder = (ctx) => {
  const worst = [...ctx.dq.rows].sort((a, b) => b.flat - a.flat)[0];
  const latencyRows = ctx.dq.rows.filter((row) => row.latency !== null);
  const latencySummary = latencyRows.length
    ? `${nf(latencyRows.reduce((a, row) => a + (row.latency ?? 0), 0) / latencyRows.length, 1)} s`
    : "no disponible (sin timestamp fiable de origen)";
  return {
    intent: "calidad",
    answer: `Calidad de telemetría ${ctx.dq.grade} (${ctx.dq.overall}/100). ${ctx.dq.verdict}`,
    bullets: [
      `Variable con más señal plana: ${VAR_META[worst.key].tag} (${nf(worst.flat, 1)} % de la ventana)`,
      `Spikes totales en ventana: ${ctx.dq.rows.reduce((a, r) => a + r.spikes, 0)}`,
      `Latencia media de tags: ${latencySummary}`,
    ],
  };
};

const ansRecomendaciones: Builder = (ctx) => {
  const recs = ctx.recs;
  return {
    intent: "recomendaciones",
    answer: recs.length
      ? `Hay ${recs.length} recomendación(es) activa(s); la de mayor prioridad es ${recs[0].prio}: «${recs[0].text}»`
      : "Sin acciones requeridas: operación estable, próxima evaluación automática en 5 min.",
    bullets: recs.slice(1, 4).map((r) => `[${r.prio}] ${r.text}`),
  };
};

const ansFlota: Builder = (ctx) => {
  const orden = [...ctx.fleet].sort((a, b) => b.score - a.score);
  const worst = orden[0];
  const fuera = ctx.fleet.filter((w) => w.level !== "ÓPTIMO").length;
  return {
    intent: "flota",
    answer: fuera
      ? `El pozo más crítico de la flota es ${worst.id} (${worst.level}, índice ${worst.score}) con caudal ${nf(worst.qNow)} Mscf/d; ${fuera} de ${ctx.fleet.length} pozos están fuera de ÓPTIMO.`
      : `Flota estable: los ${ctx.fleet.length} pozos operan en ÓPTIMO. El mayor índice es ${worst.id} (${worst.score}).`,
    bullets: orden.slice(0, 5).map((w) => `${w.id} · ${w.level} (${w.score}) · ${nf(w.qNow)} Mscf/d`),
  };
};

const ansProyeccion: Builder = (ctx) => {
  const bullets = ctx.proj.map((p) => `${p.condition}: probabilidad ${pct(p.prob)}${p.hoursEst !== null ? ` · cruce estimado en ${nf(p.hoursEst, 1)} h` : " · sin cruce proyectado en 24 h"}`);
  const risky = [...ctx.proj].sort((a, b) => b.prob - a.prob)[0];
  return {
    intent: "proyeccion",
    answer: `Proyección operacional a 24 h: el riesgo más alto es «${risky.condition}» con ${pct(risky.prob)} de probabilidad${risky.hoursEst !== null ? ` y cruce estimado en ${nf(risky.hoursEst, 1)} h` : ""}.`,
    bullets,
  };
};

const ansResumen: Builder = (ctx) => narrativeReport(ctx);

const ansAyuda: Builder = (ctx) => ({
  intent: "ayuda",
  answer:
    `Soy el copiloto de ${ctx.well.id}: consultame en lenguaje natural sobre el estado del pozo, el diagnóstico y sus evidencias, el índice de anomalía y sus drivers, la falla estimada (RUL), la declinación y EUR, la medición virtual, el choke óptimo (setpoints), el gemelo digital, la calidad de telemetría, los eventos de la flota y las recomendaciones.`,
  bullets: SUGGESTED_QUESTIONS.slice(0, 5),
});

const ansFallback: Builder = (_ctx, q) => ({
  intent: "fallback",
  answer: `No tengo seguro qué me preguntaste con «${q.trim()}». Reformulá con una de estas consultas:`,
  bullets: SUGGESTED_QUESTIONS.slice(0, 5),
});

// --------------------------- tabla de intenciones ---------------------------

interface IntentDef {
  id: string;
  kw: string[][]; // grupos: cada grupo con palabra clave encontrada suma 1
  build: Builder;
}

const INTENTS: IntentDef[] = [
  {
    id: "setpoints",
    kw: [["setpoint", "setpoints", "choke optimo", "optimo", "optimizar", "apertura", "asesor", "recomienda el choke", "mejor apertura"]],
    build: ansSetpoints,
  },
  {
    id: "gemelo",
    kw: [["gemelo", "twin", "digital twin", "desalineado", "alineado"]],
    build: ansGemelo,
  },
  {
    id: "rul",
    kw: [["rul", "vida util", "falla estimada", "cuanto queda", "cuando falla", "tiempo de falla", "weibull", "prognosis", "cuando se estima", "estima la falla"]],
    build: ansRul,
  },
  {
    id: "declinacion",
    kw: [["eur", "declinacion", "arps", "reserva", "reservas", "dca", "remanente"]],
    build: ansDeclinacion,
  },
  {
    id: "medidor",
    kw: [["medidor", "medicion virtual", "virtual", "cd", "divergencia", "bean", "calibracion"]],
    build: ansMedidor,
  },
  {
    id: "anomalia",
    kw: [["anomalia", "indice", "score", "por que subio", "drivers", "explica", "explicabilidad", "anomaly"]],
    build: ansAnomalia,
  },
  {
    id: "presion",
    kw: [["presion", "psi", "tubing", "casing", "linea"]],
    build: ansPresion,
  },
  {
    id: "caudal",
    kw: [["caudal", "produccion", "mscf", "fluido", "gas"]],
    build: ansCaudal,
  },
  {
    id: "diagnostico",
    kw: [["diagnostico", "que pasa", "que le pasa", "falla", "problema", "hipotesis", "diagnosis"]],
    build: ansDiagnostico,
  },
  {
    id: "eventos",
    kw: [["evento", "eventos", "alarma", "alarmas", "historial", "registro", "log"]],
    build: ansEventos,
  },
  {
    id: "calidad",
    kw: [["calidad", "telemetria", "sensor", "sensores", "datos", "dq"]],
    build: ansCalidad,
  },
  {
    id: "recomendaciones",
    kw: [["recomendacion", "recomendaciones", "acciones", "que hago", "acciones sugeridas", "plan"]],
    build: ansRecomendaciones,
  },
  {
    id: "flota",
    kw: [["flota", "peor pozo", "peor", "comparar", "comparativa", "todos los pozos", "fleet", "campos"]],
    build: ansFlota,
  },
  {
    id: "proyeccion",
    kw: [["proyeccion", "pronostico", "riesgo", "umbral", "probabilidad", "24 horas", "forecast"]],
    build: ansProyeccion,
  },
  {
    id: "resumen",
    kw: [["resumen", "reporte", "narrativo", "narrativa", "informe", "summary"]],
    build: ansResumen,
  },
  {
    id: "ayuda",
    kw: [["ayuda", "que puedes", "como funcionas", "help", "opciones"]],
    build: ansAyuda,
  },
  {
    id: "estado",
    kw: [["estado", "como esta", "situacion", "status", "todo bien", "panorama", "condicion"]],
    build: ansEstado,
  },
];

/**
 * Interpreta la consulta y devuelve la respuesta determinista del copiloto.
 * Empate de score → gana la intención declarada antes (orden de prioridad).
 */
export function answerQuestion(query: string, ctx: CopilotContext): CopilotAnswer {
  const q = normalizeQ(query);
  if (!q) return ansFallback(ctx, query);
  let best: { def: IntentDef; score: number } | null = null;
  for (const def of INTENTS) {
    let score = 0;
    for (const group of def.kw) for (const k of group) if (hasWord(q, k)) { score++; break; }
    if (score > 0 && (!best || score > best.score)) best = { def, score };
  }
  if (!best) return ansFallback(ctx, query);
  try {
    return best.def.build(ctx, q);
  } catch {
    return {
      intent: "error",
      answer: "No pude componer la respuesta con los datos actuales del pozo (serie insuficiente). Reintentá en unos minutos.",
      bullets: [],
    };
  }
}

/**
 * Reporte narrativo completo (intent "resumen"): párrafos que un supervisor
 * puede leer en el turno sin abrir cada tarjeta. Determinista y exportable.
 */
export function narrativeReport(ctx: CopilotContext): CopilotAnswer {
  const last = ctx.samples[ctx.samples.length - 1];
  const estado = ansEstado(ctx, "");
  const diag = ansDiagnostico(ctx, "");
  const ex = explainOf(ctx);
  const r = rulSummary(ctx);
  const dca = dcaSummary(ctx);
  const vm = vmSummary(ctx);
  const risky = [...ctx.proj].sort((a, b) => b.prob - a.prob)[0];
  const recs = ctx.recs;

  const paragraphs = [
    `1 · ESTADO — ${estado.answer}`,
    `2 · DIAGNÓSTICO — ${diag.answer}`,
    `3 · EXPLICABILIDAD — ${ex.summary}`,
    `4 · RIESGO 24 H — ${risky ? `Máxima probabilidad ${pct(risky.prob)} de ${risky.condition}${risky.hoursEst !== null ? `, cruce estimado en ${nf(risky.hoursEst, 1)} h` : ""}.` : "Sin umbrales en riesgo."}`,
    `5 · VIDA ÚTIL — ${r.mode !== "normal" ? `Falla estimada ${fmtHours(r.rulMedian)} (p10–p90: ${fmtHours(r.p10)} – ${fmtHours(r.p90)}) por ${modeLabel(r.mode)}.` : "Sin episodio de degradación activo."}`,
    `6 · DECLINACIÓN — ${dca.fit.feasible ? `Ajuste ${dca.fit.model} (R² ${nf(dca.fit.r2, 3)}), Di efectiva ${nf(dca.monthly ?? 0, 1)} %/mes, EUR ${nf(dca.fit.eur / 1000, 1)} MMscf.` : `Ajuste no defendible (${dca.fit.reason ?? "serie no explicada"}).`}`,
    `7 · MEDICIÓN VIRTUAL — ${vm.chk.cdCalibrated !== null && vm.chk.qVirtual > 1 ? `Q virtual ${nf(vm.chk.qVirtual)} vs medidor ${nf(vm.chk.qMeasured)} Mscf/d (Δ ${SIGN(vm.chk.devPct)}${nf(vm.chk.devPct, 1)} %, ${vm.chk.flag ? "fuera" : "dentro"} de tolerancia).` : "Calibración no defendible en condiciones actuales."}`,
    `8 · ACCIONES — ${recs.length ? recs.map((x) => `[${x.prio}] ${x.text}`).join(" ") : "Sin acciones requeridas."}`,
  ];

  return {
    intent: "resumen",
    answer: `REPORTE NARRATIVO · ${ctx.well.id} · ${nf(last.q)} Mscf/d · índice ${ctx.anom.score}/100 (${ctx.anom.level})\n${paragraphs.join("\n\n")}`,
    bullets: [
      `Calidad de telemetría ${ctx.dq.grade} (${ctx.dq.overall}/100)`,
      `${ctx.fleet.filter((w) => w.level !== "ÓPTIMO").length} de ${ctx.fleet.length} pozos fuera de ÓPTIMO`,
    ],
  };
}

