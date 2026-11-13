import { useEffect, useMemo, useRef, useState } from "react";
import { SCENARIO_INFO, VAR_KEYS, VAR_META, createFleet, fmtClock, mergeEvents, sanitizeSample } from "./lib/sim";
import type { Sample, Scenario, VarKey, WellEvent, WellSim } from "./lib/sim";
import { OpcuaBridgeSource } from "./lib/ingest/opcuaBridge";
import { SourceSampleAccumulator } from "./lib/ingest/sampleClock";
import type { LinkStatus, LinkStats, TagMap } from "./lib/ingest/opcuaBridge";
import { anomaly, anomalyLevel, dataQuality, diagnose, projections, recommend } from "./lib/models";
import type {
  AnomalyResult,
  DataQuality,
  Hypothesis,
  ProjRow,
  Recommendation,
} from "./lib/models";
import { explainSamples } from "./lib/explain";
import type { ExplainResult } from "./lib/explain";
import type { CopilotContext } from "./lib/copilot";
import { mergeDiagnosis } from "./lib/ml/engine";
import { VigiaEngine } from "./lib/ml/engine";
import type { ClassProb, EngineLike, MlAnomaly, MlForecast, TrainState } from "./lib/ml/engine";
import { createEngine } from "./lib/ml/engineProxy";
import { buildWellReport, downloadWellCsv, downloadWellReportJson, printWellReport } from "./lib/export";
import { fleetCompare } from "./lib/fleet";
import type { FleetRow } from "./lib/fleet";
import { ComparePanel } from "./components/ComparePanel";
import { ArpsCard } from "./components/ArpsCard";
import { RulCard } from "./components/RulCard";
import { VirtualMeterCard } from "./components/VirtualMeterCard";
import { SetpointAdvisorCard } from "./components/SetpointAdvisorCard";
import { TwinCard } from "./components/TwinCard";
import { ExplainCard } from "./components/ExplainCard";
import { CopilotPanel } from "./components/CopilotPanel";
import { SourcePanel } from "./components/SourcePanel";
import { TopBar } from "./components/TopBar";
import { WellRail } from "./components/WellRail";
import type { WellSummary } from "./components/WellRail";
import { KpiGrid } from "./components/KpiGrid";
import { ForecastChart } from "./components/ForecastChart";
import { AnomalyPanel, DataQualityPanel, ScenarioControls } from "./components/InsightPanels";
import { DiagnosisPanel, EventLog, ProjectionPanel, RecommendationsPanel } from "./components/RightRail";
import { TrainingPanel } from "./components/TrainingPanel";
import { StatusChip } from "./components/bits";

interface View {
  summaries: WellSummary[];
  sel: WellSim;
  samples: Sample[];
  anom: AnomalyResult;
  anomMl: boolean;
  diag: Hypothesis[];
  diagMl: boolean;
  proj: ProjRow[];
  recs: Recommendation[];
  dq: DataQuality;
  explain: ExplainResult;
  events: WellEvent[];
  alarmCount: number;
  observing: number;
}

interface MlWellResult {
  anom: MlAnomaly;
  probs: ClassProb[];
}

function refreshLevels(wells: WellSim[], log: boolean) {
  for (const w of wells) {
    const a = anomaly(w.buf.slice(-240));
    const prev = w.level;
    w.score = a.score;
    w.level = a.level;
    if (log && prev !== a.level) {
      w.log(
        a.level === "ÓPTIMO" ? "ok" : a.level === "VIGILAR" ? "warn" : "alarm",
        `Índice de anomalía ${a.level === "ÓPTIMO" ? "bajó" : "subió"} a ${a.level} (${a.score})`
      );
    }
  }
}

function buildView(
  wells: WellSim[],
  selId: string,
  mlCache: Map<string, MlWellResult> | null,
  mlFc: MlForecast | null,
): View {
  const sel = wells.find((w) => w.id === selId) ?? wells[0];
  const summaries: WellSummary[] = wells.map((w) => ({
    id: w.id,
    name: w.name,
    field: w.field,
    status: w.level,
    score: w.score,
    qNow: w.last.q,
    spark: w.buf.slice(-48).map((s) => s.q),
  }));
  const samples = sel.buf.slice(-240);
  const mlSel = mlCache?.get(sel.id) ?? null;
  const anom: AnomalyResult = mlSel
    ? {
        score: mlSel.anom.score,
        level: mlSel.anom.level,
        contributions: mlSel.anom.contributions,
      }
    : anomaly(sel.buf.slice(-240));
  let diag = diagnose(sel.buf.slice(-160));
  let diagMl = false;
  if (mlSel && mlSel.probs.length > 0) {
    diag = mergeDiagnosis(diag, mlSel.probs, sel.buf.slice(-160));
    diagMl = true;
  }
  const proj = projections(samples, sel.base, mlFc);
  const recs = recommend(diag, proj, samples);
  const dq = dataQuality(sel.buf.slice(-260));
  const explain = explainSamples(samples, { score: anom.score, level: anom.level });
  const events = mergeEvents(wells, 42);
  const alarmCount = wells.filter((w) => w.level === "ALERTA" || w.level === "CRÍTICO").length;
  const observing = wells.filter((w) => w.level !== "ÓPTIMO").length;
  return {
    summaries,
    sel,
    samples,
    anom,
    anomMl: !!mlSel,
    diag,
    diagMl,
    proj,
    recs,
    dq,
    explain,
    events,
    alarmCount,
    observing,
  };
}

export default function App() {
  const fleetRef = useRef<WellSim[] | null>(null);
  if (!fleetRef.current) {
    fleetRef.current = createFleet();
    refreshLevels(fleetRef.current, false);
  }

  const [selId, setSelId] = useState("PN-041");
  const [paused, setPaused] = useState(false);
  const [horizon, setHorizon] = useState(12);
  const [varKey, setVarKey] = useState<VarKey>("pt");
  const [doneRecs, setDoneRecs] = useState<Set<string>>(new Set());
  const [tickN, setTickN] = useState(0);

  // ------------------------- motor ML (TensorFlow.js) -----------------------
  // corre en un Web Worker (EngineProxy); fallback automático al hilo
  // principal (VigiaEngine) en entornos sin soporte de module workers
  const engineRef = useRef<EngineLike | null>(null);
  const mlCacheRef = useRef<Map<string, MlWellResult>>(new Map());
  const mlFcRef = useRef<MlForecast | null>(null);
  const [mlState, setMlState] = useState<TrainState>(VigiaEngine.initialState());
  const [mlReady, setMlReady] = useState(false);

  const startTraining = () => {
    // motor anterior: abort + dispose (el proxy termina el worker viejo:
    // solo abort dejaría el hilo del worker vivo = fuga de memoria)
    engineRef.current?.abort();
    engineRef.current?.dispose();
    const eng = createEngine();
    engineRef.current = eng;
    mlCacheRef.current = new Map();
    mlFcRef.current = null;
    setMlReady(false);
    setMlState(VigiaEngine.initialState());
    eng
      .trainAll((s) => {
        if (engineRef.current === eng) setMlState(s); // ignora motores obsoletos
      })
      .then((ok) => {
        if (engineRef.current === eng) setMlReady(ok);
      })
      .catch(() => {
        if (engineRef.current === eng) setMlReady(false);
      });
  };

  const cancelTraining = () => engineRef.current?.abort();

  useEffect(() => {
    startTraining();
    return () => {
      engineRef.current?.dispose();
      engineRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ------------------------------ ciclo de 1.5 s ----------------------------
  useEffect(() => {
    let alive = true;
    let tickCount = 0;
    let busy = false; // guardia de reentrancia: si la inferencia tarda más de
    // un ciclo (p. ej. backend CPU), el próximo tick no la solapa
    const step = async () => {
      if (busy) return;
      busy = true;
      try {
        const wells = fleetRef.current!;
        if (!paused) wells.forEach((w) => w.tick());

        // niveles de flota: si hay caché ML se mantiene (evita parpadeo stats↔ML)
        const cache = mlCacheRef.current;
        for (const w of wells) {
          const r = cache.get(w.id);
          const a = r
            ? { score: r.anom.score, level: anomalyLevel(r.anom.score) }
            : anomaly(w.buf.slice(-240));
          const prev = w.level;
          w.score = a.score;
          w.level = a.level;
          if (prev !== a.level) {
            w.log(
              a.level === "ÓPTIMO" ? "ok" : a.level === "VIGILAR" ? "warn" : "alarm",
              `${r ? "[ML] " : ""}Índice de anomalía ${a.level === "ÓPTIMO" ? "bajó" : "subió"} a ${a.level} (${a.score})`
            );
          }
        }

        // inferencia ML cada 2 ciclos (3 s): suficiente y sin saturar la GPU
        const eng = engineRef.current;
        const infer = tickCount % 2 === 0;
        tickCount++;
        if (eng?.ready && infer) {
          try {
            const sel = wells.find((w) => w.id === selId) ?? wells[0];
            const [results, fc] = await Promise.all([
              Promise.all(
                wells.map(async (w) => ({ id: w.id, r: await eng.assess(w.buf, w.base) })),
              ),
              eng.forecastSeries(sel.buf, sel.base, 96),
            ]);
            if (!alive) return;
            const map = new Map<string, MlWellResult>();
            for (const { id, r } of results) if (r) map.set(id, r);
            mlCacheRef.current = map;
            mlFcRef.current = fc;
            // el score de flota pasa a medirse con el autoencoder
            for (const w of wells) {
              const r = map.get(w.id);
              if (!r) continue;
              const prev = w.level;
              w.score = r.anom.score;
              w.level = anomalyLevel(r.anom.score);
              if (prev !== w.level) {
                w.log(
                  w.level === "ÓPTIMO" ? "ok" : w.level === "VIGILAR" ? "warn" : "alarm",
                  `[ML] Índice de anomalía ${w.level === "ÓPTIMO" ? "bajó" : "subió"} a ${w.level} (${w.score}) · autoencoder`
                );
              }
            }
          } catch {
            /* inferencia fallida → se mantiene el pipeline estadístico */
          }
        }
        if (alive) setTickN((n) => n + 1);
      } finally {
        busy = false;
      }
    };
    const id = setInterval(() => void step(), 1500);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [paused, mlReady, selId]);

  const view = useMemo(
    () => buildView(fleetRef.current!, selId, mlCacheRef.current, mlFcRef.current),
    [tickN, selId],
  );

  // comparativa de flota: se recalcula con el mismo ritmo que la vista
  const fleetRows = useMemo<FleetRow[]>(
    () => fleetCompare(fleetRef.current!),
    [tickN],
  );

  const topDiagId = view.diag[0]?.id ?? "";
  useEffect(() => {
    setDoneRecs(new Set());
  }, [selId, topDiagId]);

  const thrLow = useMemo(() => {
    const b = view.sel.base;
    if (varKey === "pt") return b.pt * 0.82;
    if (varKey === "q") return b.q * 0.78;
    if (varKey === "pl") return b.pl * 0.85;
    return undefined;
  }, [view.sel, varKey]);

  // ------------------------ fuente externa (puente OPC-UA) ------------------
  // las muestras del puente entran saneadas al buffer del pozo seleccionado;
  // el simulador queda en pausa mientras la fuente externa está activa
  const sourceRef = useRef<OpcuaBridgeSource | null>(null);
  const sourceAccumulatorRef = useRef<SourceSampleAccumulator | null>(null);
  const sourceWellIdRef = useRef<string | null>(null);
  const selIdRef = useRef(selId);
  selIdRef.current = selId;
  const [srcUrl, setSrcUrl] = useState("ws://localhost:8082");
  const [srcStatus, setSrcStatus] = useState<LinkStatus>("idle");
  const [srcStats, setSrcStats] = useState<LinkStats>({ received: 0, accepted: 0, rejected: 0, reconnects: 0 });
  const [srcInfo, setSrcInfo] = useState("");

  const connectSource = () => {
    disconnectSource();
    const src = new OpcuaBridgeSource();
    sourceAccumulatorRef.current = new SourceSampleAccumulator();
    sourceWellIdRef.current = selIdRef.current;
    src.onData((s) => {
      const wells = fleetRef.current!;
      const w = wells.find((x) => x.id === selIdRef.current) ?? wells[0];
      const accumulator = sourceAccumulatorRef.current;
      const update = accumulator?.ingest(s) ?? null;
      const changedWell = sourceWellIdRef.current !== w.id;
      if (changedWell) sourceWellIdRef.current = w.id;

      if (changedWell || update?.kind === "initial") {
        const snapshot = update?.sample ?? accumulator?.current() ?? null;
        if (snapshot) {
          // Nunca mezclar historial sintético con telemetría real.
          w.buf.splice(0, w.buf.length, snapshot);
          w.base = {
            pt: snapshot.pt, pc: snapshot.pc, pl: snapshot.pl,
            temp: snapshot.temp, q: snapshot.q, choke: snapshot.choke,
          };
          w.m = snapshot.m;
          w.setScenario("normal");
          w.log("warn", "Fuente OPC-UA: buffer reiniciado; línea base provisional tomada de la primera muestra completa. Validar baseline con ingeniería.");
          refreshLevels(wells, true);
          setTickN((n) => n + 1);
        }
      } else if (update?.kind === "new-minute") {
        w.buf.push(update.sample);
        if (w.buf.length > 520) w.buf.shift();
        refreshLevels(wells, true);
        setTickN((n) => n + 1);
      } else if (update?.kind === "update" && w.buf.length) {
        // Seis tags pueden llegar en frames distintos dentro del mismo minuto:
        // se combinan en una muestra, no se cuentan como seis minutos.
        w.buf[w.buf.length - 1] = update.sample;
        refreshLevels(wells, true);
        setTickN((n) => n + 1);
      }
      setSrcStats(src.stats);
    });
    src.onStatus((st, info) => {
      setSrcStatus(st);
      setSrcInfo(info);
      setSrcStats(src.stats);
    });
    // mapa por defecto: los tags de demostración del propio consola
    const tags: TagMap = {
      pt: VAR_META.pt.tag,
      pc: VAR_META.pc.tag,
      pl: VAR_META.pl.tag,
      temp: VAR_META.temp.tag,
      q: VAR_META.q.tag,
      choke: VAR_META.choke.tag,
    };
    src.connect(srcUrl, tags);
    sourceRef.current = src;
    setPaused(true);
    const sel = fleetRef.current!.find((x) => x.id === selIdRef.current) ?? fleetRef.current![0];
    sel.log("info", `Fuente externa conectada: ${srcUrl} (simulador en pausa)`);
    refreshLevels(fleetRef.current!, true);
  };

  const disconnectSource = () => {
    sourceAccumulatorRef.current = null;
    sourceWellIdRef.current = null;
    if (sourceRef.current) {
      sourceRef.current.disconnect();
      sourceRef.current = null;
    }
    setSrcStatus("idle");
    setSrcInfo("");
    setSrcStats({ received: 0, accepted: 0, rejected: 0, reconnects: 0 });
  };

  useEffect(() => {
    return () => {
      sourceRef.current?.disconnect();
      sourceRef.current = null;
    };
  }, []);

  const inject = (s: Scenario) => {
    view.sel.setScenario(s);
    refreshLevels(fleetRef.current!, true);
    setTickN((n) => n + 1);
  };

  const exportCsv = () => {
    downloadWellCsv(view.sel.id, fleetRef.current!.find((w) => w.id === view.sel.id)?.buf ?? view.samples);
  };

  const exportReport = () => {
    const report = buildWellReport({
      well: { id: view.sel.id, name: view.sel.name, field: view.sel.field, depth: view.sel.depth },
      samples: view.samples,
      anom: view.anom,
      anomMl: view.anomMl,
      diag: view.diag,
      proj: view.proj,
      recs: view.recs,
      dq: view.dq,
      events: view.events,
    });
    downloadWellReportJson(report);
  };

  const exportPdf = () => {
    printWellReport(buildWellReport({
      well: { id: view.sel.id, name: view.sel.name, field: view.sel.field, depth: view.sel.depth },
      samples: view.samples,
      anom: view.anom,
      anomMl: view.anomMl,
      diag: view.diag,
      proj: view.proj,
      recs: view.recs,
      dq: view.dq,
      events: view.events,
    }));
  };

  // atajos de teclado: 1..5 selecciona pozo · P pausa · C cicla variable
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      const wells = fleetRef.current!;
      const n = Number(e.key);
      if (!Number.isNaN(n) && n >= 1 && n <= wells.length) {
        setSelId(wells[n - 1].id);
      } else if (e.key === "p" || e.key === "P") {
        setPaused((p) => !p);
      } else if (e.key === "c" || e.key === "C") {
        setVarKey((k) => VAR_KEYS[(VAR_KEYS.indexOf(k) + 1) % VAR_KEYS.length]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const toggleRec = (id: string) => {
    setDoneRecs((prev) => {
      const next = new Set(prev);
      if (prev.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // contexto del copiloto: la misma vista que consume la UI, reconstruida con
  // el mismo ritmo del tick — el chat siempre responde sobre el estado vivo
  const copilotCtx = useMemo<CopilotContext>(
    () => ({
      well: { id: view.sel.id, name: view.sel.name, field: view.sel.field, depth: view.sel.depth },
      samples: view.samples,
      base: view.sel.base,
      anom: { score: view.anom.score, level: view.anom.level, contributions: view.anom.contributions, ml: view.anomMl },
      diag: view.diag,
      diagMl: view.diagMl,
      proj: view.proj,
      recs: view.recs,
      dq: view.dq,
      events: view.events,
      fleet: view.summaries.map((s) => ({ id: s.id, name: s.name, level: s.status, score: s.score, qNow: s.qNow })),
    }),
    [view],
  );

  return (
    <div className="relative z-10 min-h-screen flex flex-col font-body">
      <TopBar
        simM={view.sel.last.m}
        paused={paused}
        onTogglePause={() => setPaused((p) => !p)}
        alarmCount={view.alarmCount}
        latestEvent={view.events[0] ?? null}
        observing={view.observing}
        ml={{ phase: mlState.phase, progress: mlState.progress }}
      />

      <main className="flex-1 w-full max-w-[1600px] mx-auto px-3 md:px-4 py-3 grid grid-cols-12 gap-3 items-start">
        {/* flota */}
        <div className="col-span-12 md:col-span-4 xl:col-span-3 md:sticky md:top-3">
          <WellRail wells={view.summaries} selId={view.sel.id} onSelect={setSelId} />
        </div>

        {/* consola central */}
        <section className="col-span-12 md:col-span-8 xl:col-span-6 flex flex-col gap-3 min-w-0">
          {/* encabezado del pozo */}
          <div className="panel corners p-3.5 flex flex-wrap items-center gap-x-5 gap-y-2 rise">
            <div>
              <div className="flex items-center gap-3">
                <h1 className="font-display font-bold text-[30px] md:text-[34px] tracking-[0.06em] leading-none text-fg">
                  {view.sel.id}
                </h1>
                <StatusChip status={view.sel.level} />
              </div>
              <p className="text-[11px] text-fg3 mt-1.5">
                {view.sel.name} · {view.sel.field} · TVD {view.sel.depth}
              </p>
            </div>
            <div className="ml-auto flex items-center gap-4 font-mono text-[9.5px] tracking-[0.08em] text-fg3">
              <span>
                RÉGIMEN <span className="text-fg2">{SCENARIO_INFO[view.sel.scenario].label.toUpperCase()}</span>
              </span>
              <span>
                ÚLTIMA MUESTRA <span className="text-fg2">{fmtClock(view.sel.last.m)}</span>
              </span>
              <span className="hidden sm:flex items-center gap-1.5 text-ok">
                <span className={`w-1.5 h-1.5 rounded-full ${paused ? "bg-watch" : "bg-ok live-dot"}`} />
                STREAM 1 MIN
              </span>
              <span className="flex items-center gap-1.5">
                <button
                  onClick={exportCsv}
                  aria-label="Descargar telemetría completa del pozo en CSV"
                  title="Descargar telemetría completa del pozo en CSV"
                  className="px-2 py-1 rounded border border-line text-fg2 hover:border-ok/60 hover:text-ok transition-colors"
                >
                  CSV
                </button>
                <button
                  onClick={exportPdf}
                  aria-label="Generar reporte operativo imprimible (PDF)"
                  title="Generar reporte operativo imprimible (PDF)"
                  className="px-2 py-1 rounded border border-line text-fg2 hover:border-ok/60 hover:text-ok transition-colors"
                >
                  PDF
                </button>
                <button
                  onClick={exportReport}
                  aria-label="Descargar reporte operativo completo en JSON"
                  title="Descargar reporte operativo completo en JSON"
                  className="px-2 py-1 rounded border border-line text-fg2 hover:border-ok/60 hover:text-ok transition-colors"
                >
                  JSON
                </button>
              </span>
            </div>
          </div>

          <KpiGrid samples={view.samples} />

          <ComparePanel rows={fleetRows} selId={view.sel.id} onSelect={setSelId} />

          <ForecastChart
            samples={view.samples}
            varKey={varKey}
            onVarKey={setVarKey}
            horizon={horizon}
            onHorizon={setHorizon}
            thrLow={thrLow}
            base={view.sel.base}
            engine={engineRef.current}
            engineReady={mlReady}
          />

          <ArpsCard wellId={view.sel.id} baseQ={view.sel.base.q} />

          <div className="flex flex-wrap gap-3">
            <RulCard diag={view.diag} samples={view.samples} base={view.sel.base} anomScore={view.anom.score} />
            <VirtualMeterCard samples={view.samples} base={view.sel.base} />
          </div>

          <div className="flex flex-wrap gap-3">
            <SetpointAdvisorCard diag={view.diag} samples={view.samples} base={view.sel.base} anomScore={view.anom.score} />
            <TwinCard samples={view.samples} />
          </div>

          <div className="flex flex-wrap gap-3">
            <AnomalyPanel result={view.anom} ml={view.anomMl} />
            <DataQualityPanel dq={view.dq} />
          </div>

          <ExplainCard ex={view.explain} />

          <ScenarioControls onInject={inject} active={view.sel.scenario} />

          <SourcePanel
            status={srcStatus}
            stats={srcStats}
            info={srcInfo}
            url={srcUrl}
            onUrl={setSrcUrl}
            onConnect={connectSource}
            onDisconnect={disconnectSource}
          />

          <TrainingPanel state={mlState} onRetrain={startTraining} onCancel={cancelTraining} />
        </section>

        {/* inteligencia */}
        <aside className="col-span-12 xl:col-span-3 flex flex-col gap-3 min-w-0">
          <CopilotPanel ctx={copilotCtx} />
          <DiagnosisPanel diag={view.diag} ml={view.diagMl} />
          <ProjectionPanel proj={view.proj} />
          <RecommendationsPanel recs={view.recs} done={doneRecs} onToggle={toggleRec} />
          <EventLog events={view.events} />
        </aside>
      </main>

      <footer className="relative z-10 border-t border-line mt-1">
        <div className="max-w-[1600px] mx-auto px-3 md:px-4 py-2.5 flex flex-wrap items-center gap-x-5 gap-y-1 font-mono text-[9px] tracking-[0.08em] text-fg3">
          <span>
            MODELOS EN NAVEGADOR · TENSORFLOW.JS · LSTM N1 + AUTOENCODER N2 + CLASIFICADOR N3 ·
            FALLBACK ESTADÍSTICO (HOLT / Z-SCORE / REGLAS v2.4)
          </span>
          <span className="hidden md:inline">DCA ARPS · RUL WEIBULL · MEDICIÓN VIRTUAL · COPILOTO NL · EXPLICABILIDAD · ASESOR SETPOINTS · GEMELO DIGITAL</span>
          <span className="hidden lg:inline">ATAJOS: 1–5 POZO · P PAUSA · C VARIABLE</span>
          <span className="ml-auto">TELEMETRÍA SINTÉTICA CON FINES DE DEMOSTRACIÓN · VIGÍA ML v0.11 · 2026</span>
        </div>
      </footer>
    </div>
  );
}
