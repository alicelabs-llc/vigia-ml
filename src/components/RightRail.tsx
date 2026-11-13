import type { Hypothesis, ProjRow, Recommendation } from "../lib/models";
import { fmtClock } from "../lib/sim";
import type { WellEvent } from "../lib/sim";
import { HYPO_ICON, EngineChip, IconCheck, IconTarget, PrioChip, SectionHead } from "./bits";

// ----------------------------- N3 · diagnóstico ----------------------------
export function DiagnosisPanel({ diag, ml = false }: { diag: Hypothesis[]; ml?: boolean }) {
  const top = diag[0];
  const rest = diag.slice(1, 3);
  const TopIcon = top ? HYPO_ICON[top.icon] : IconCheck;
  const isNormal = top?.id === "normal";
  const accent = isNormal ? "#3fd0b6" : top && top.conf > 0.75 ? "#f26d5f" : "#f2b33d";

  return (
    <div className="panel corners panel-glow p-3.5 rise" style={{ animationDelay: "200ms" }}>
      <SectionHead level="N3" title="Diagnóstico" accent={accent} right={<EngineChip ml={ml} small />} />
      {top && (
        <div className="flex items-start gap-3">
          <span
            className={`w-9 h-9 grid place-items-center border shrink-0 ${isNormal ? "" : ""}`}
            style={{ color: accent, borderColor: `${accent}55`, background: `${accent}0d` }}
          >
            <TopIcon size={17} />
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline justify-between gap-2">
              <h4 className="font-display font-semibold text-[13.5px] leading-tight text-fg">{top.name}</h4>
              <span className="font-mono text-[12px] font-semibold tabular-nums shrink-0" style={{ color: accent }}>
                {(top.conf * 100).toFixed(0)}%
              </span>
            </div>
            <div className="mt-1.5 h-[4px] bg-[#1b2e37] overflow-hidden">
              <span className="block h-full ease-bar" style={{ width: `${top.conf * 100}%`, background: accent }} />
            </div>
            <ul className="mt-2 space-y-1">
              {top.evidence.map((e, i) => (
                <li key={i} className="flex gap-1.5 text-[10.5px] text-fg2 leading-snug">
                  <span className="text-fg3 font-mono text-[9px] mt-[2px] shrink-0">▸</span>
                  <span>{e}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
      {rest.length > 0 && (
        <div className="mt-2.5 pt-2.5 border-t border-line space-y-1">
          {rest.map((h) => {
            const Ic = HYPO_ICON[h.icon];
            return (
              <div key={h.id} className="flex items-center gap-2 text-[10.5px] text-fg3">
                <Ic size={12} className="shrink-0 text-fg3" />
                <span className="truncate flex-1">{h.name}</span>
                <span className="font-mono tabular-nums shrink-0">{(h.conf * 100).toFixed(0)}%</span>
                <span className="w-10 h-[3px] bg-[#1b2e37] overflow-hidden shrink-0">
                  <span className="block h-full ease-bar bg-fg3/60" style={{ width: `${h.conf * 100}%` }} />
                </span>
              </div>
            );
          })}
        </div>
      )}
      <p className="mt-2.5 font-mono text-[8.5px] text-fg3/80 tracking-[0.06em]">
        {ml
          ? "RED NEURONAL v1 · SOFTMAX 5 CLASES · FUSIONADA CON REGLAS v2.4"
          : "REGLAS v2.4 · 23 PATRONES · VENTANA 45 MIN"}
      </p>
    </div>
  );
}

// ------------------------- N4 · proyección operacional ---------------------
export function ProjectionPanel({ proj }: { proj: ProjRow[] }) {
  return (
    <div className="panel corners panel-glow p-3.5 rise" style={{ animationDelay: "260ms" }}>
      <SectionHead level="N4" title="Proyección operacional" accent="#7fb4e6" />
      <div className="space-y-2.5">
        {proj.map((p) => {
          const pct = Math.round(p.prob * 100);
          const c = p.prob > 0.6 ? "#f26d5f" : p.prob > 0.35 ? "#f2b33d" : "#3fd0b6";
          return (
            <div key={p.key}>
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-[10.5px] text-fg2">{p.condition}</span>
                <span className="font-mono text-[10px] tabular-nums shrink-0" style={{ color: c }}>
                  {p.hoursEst !== null ? `~${p.hoursEst.toFixed(0)} h` : "> 24 h"}
                </span>
              </div>
              <div className="mt-1 flex items-center gap-2">
                <span className="flex-1 h-[5px] bg-[#1b2e37] overflow-hidden">
                  <span className="block h-full ease-bar" style={{ width: `${pct}%`, background: c }} />
                </span>
                <span className="font-mono text-[11px] font-semibold tabular-nums w-9 text-right" style={{ color: c }}>
                  {pct}%
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-2.5 text-[9.5px] text-fg3 leading-snug">
        “Si mantenemos esta configuración, ¿cuándo se alcanza la condición?” — Monte Carlo analítico
        sobre el pronóstico {" "}N1 (LSTM o Holt según el motor activo).
      </p>
    </div>
  );
}

// --------------------------- N5 · recomendaciones --------------------------
export function RecommendationsPanel({
  recs,
  done,
  onToggle,
}: {
  recs: Recommendation[];
  done: Set<string>;
  onToggle: (id: string) => void;
}) {
  return (
    <div className="panel corners panel-glow p-3.5 rise" style={{ animationDelay: "320ms" }}>
      <SectionHead level="N5" title="Recomendaciones" accent="#f2873d" />
      <ol className="space-y-2">
        {recs.map((r, i) => {
          const isDone = done.has(r.id);
          return (
            <li
              key={r.id}
              className={`group border border-line bg-[#0e1a20] px-2.5 py-2 transition-all duration-200 hover:border-line2 ${
                isDone ? "opacity-45" : ""
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="font-display font-bold text-[13px] text-fg3 w-4 shrink-0">{i + 1}</span>
                <PrioChip prio={r.prio} />
                <button
                  onClick={() => onToggle(r.id)}
                  className={`ml-auto inline-flex items-center gap-1 font-mono text-[8.5px] tracking-[0.1em] px-1.5 py-[3px] border transition-colors ${
                    isDone
                      ? "text-ok border-[#3fd0b655]"
                      : "text-fg3 border-transparent hover:text-fg2 hover:border-line"
                  }`}
                >
                  <IconCheck size={10} />
                  {isDone ? "REVISADA" : "MARCAR REVISIÓN"}
                </button>
              </div>
              <p className={`mt-1.5 text-[11px] leading-snug text-fg2 ${isDone ? "line-through" : ""}`}>{r.text}</p>
            </li>
          );
        })}
      </ol>
      <p className="mt-2.5 font-mono text-[8.5px] text-fg3/80 tracking-[0.06em]">
        SOLO ORIENTATIVAS · NO EJECUTAN ACCIONES EN SCADA · VALIDACIÓN Y AUTORIZACIÓN DE OPERACIÓN OBLIGATORIAS
      </p>
    </div>
  );
}

// ------------------------------ eventos ------------------------------------
const SEV_COLOR: Record<string, string> = {
  info: "#5f7a82",
  ok: "#3fd0b6",
  warn: "#f2b33d",
  alarm: "#f26d5f",
};

export function EventLog({ events }: { events: WellEvent[] }) {
  return (
    <div className="panel p-3.5 rise" style={{ animationDelay: "380ms" }}>
      <SectionHead level="LOG" title="Eventos" />
      <div className="max-h-56 overflow-y-auto pr-1 space-y-[7px]">
        {events.map((e) => (
          <div key={e.id} className="flex items-start gap-2 text-[10.5px] leading-snug">
            <span className="w-1.5 h-1.5 rounded-full mt-[4px] shrink-0" style={{ background: SEV_COLOR[e.severity] }} />
            <span className="font-mono text-[9px] text-fg3 shrink-0 tabular-nums">{fmtClock(e.m)}</span>
            <span className="font-mono text-[9px] shrink-0" style={{ color: SEV_COLOR[e.severity] }}>
              {e.wellId}
            </span>
            <span className="text-fg2 min-w-0">{e.msg}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
