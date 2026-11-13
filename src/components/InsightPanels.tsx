import type { AnomalyResult, DataQuality } from "../lib/models";
import { SCENARIO_INFO, VAR_META } from "../lib/sim";
import type { Scenario } from "../lib/sim";
import { EngineChip, STATUS_STYLE, IconBolt, IconCheck, IconChip, IconDroplet, IconGauge, IconValve, SectionHead } from "./bits";

// ------------------------- N2 · medidor de anomalía ------------------------
const polar = (cx: number, cy: number, r: number, aDeg: number): [number, number] => [
  cx + r * Math.cos((aDeg * Math.PI) / 180),
  cy - r * Math.sin((aDeg * Math.PI) / 180),
];
const arcPath = (cx: number, cy: number, r: number, a0: number, a1: number) => {
  const [x0, y0] = polar(cx, cy, r, a0);
  const [x1, y1] = polar(cx, cy, r, a1);
  return `M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${r} ${r} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
};

const ZONES = [
  { from: 180, to: 135, color: "#3fd0b6" },
  { from: 135, to: 81, color: "#f2b33d" },
  { from: 81, to: 36, color: "#f2873d" },
  { from: 36, to: 0, color: "#f26d5f" },
];

export function AnomalyPanel({ result, ml = false }: { result: AnomalyResult; ml?: boolean }) {
  const st = STATUS_STYLE[result.level];
  const deg = result.score * 1.8 - 90;

  return (
    <div className="panel corners panel-glow p-3.5 flex-1 min-w-[260px] rise" style={{ animationDelay: "240ms" }}>
      <SectionHead level="N2" title="Detección de anomalías" accent="#f2b33d" right={<EngineChip ml={ml} small />} />
      <div className="flex items-center gap-4">
        <svg viewBox="0 0 200 118" className="w-[168px] shrink-0">
          {ZONES.map((z, i) => (
            <path key={i} d={arcPath(100, 100, 80, z.from, z.to)} stroke={z.color} strokeWidth="7" fill="none" opacity="0.85" strokeLinecap="butt" />
          ))}
          {[25, 55, 80].map((s) => {
            const a = 180 - s * 1.8;
            const [x0, y0] = polar(100, 100, 70, a);
            const [x1, y1] = polar(100, 100, 88, a);
            return <line key={s} x1={x0} y1={y0} x2={x1} y2={y1} stroke="#0a1216" strokeWidth="2" />;
          })}
          <g style={{ transform: `rotate(${deg}deg)`, transformOrigin: "100px 100px" }} className="ease-needle">
            <line x1="100" y1="100" x2="100" y2="32" stroke={st.color} strokeWidth="2.5" strokeLinecap="round" />
            <circle cx="100" cy="100" r="5" fill={st.color} />
          </g>
          <text x="100" y="72" textAnchor="middle" fontSize="26" fontWeight="600" fill={st.color} fontFamily="IBM Plex Mono, monospace">
            {result.score}
          </text>
          <text x="100" y="112" textAnchor="middle" fontSize="8.5" fill="#5f7a82" fontFamily="IBM Plex Mono, monospace" letterSpacing="2">
            {ml ? "AUTOENCODER · 180→20→180 · 30 MIN" : "ÍNDICE 0–100 · z-score 120 min"}
          </text>
        </svg>

        <div className="min-w-0 flex-1">
          <div
            className="inline-flex items-center gap-1.5 font-mono text-[10px] font-semibold tracking-[0.14em] px-2 py-1 border"
            style={{ color: st.color, borderColor: `${st.color}66`, background: `${st.color}12` }}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${result.level === "CRÍTICO" ? "blink" : ""}`} style={{ background: st.color }} />
            {result.level}
          </div>
          <div className="mt-2.5 space-y-1.5">
            {result.contributions.length === 0 ? (
              <p className="text-[11px] text-fg3 leading-relaxed">
                Sin contribuyentes relevantes: todas las señales dentro de ±2σ de su línea base móvil.
              </p>
            ) : (
              result.contributions.map((c) => (
                <div key={c.key} className="flex items-center gap-2">
                  <span className="font-mono text-[9px] text-fg3 w-14 shrink-0">{VAR_META[c.key].tag}</span>
                  <span className="flex-1 h-[4px] bg-[#1b2e37] overflow-hidden">
                    <span
                      className="block h-full ease-bar"
                      style={{ width: `${Math.min(100, (c.z / 6) * 100)}%`, background: st.color }}
                    />
                  </span>
                  <span className="font-mono text-[10px] tabular-nums" style={{ color: st.color }}>
                    z {c.z.toFixed(1)}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ------------------------- calidad de datos --------------------------------
export function DataQualityPanel({ dq }: { dq: DataQuality }) {
  const gradeColor = dq.grade === "A" ? "#3fd0b6" : dq.grade === "B" ? "#f2b33d" : dq.grade === "C" ? "#f2873d" : "#f26d5f";
  const checklist: { t: string; state: "ok" | "pending" | "sim" }[] = [
    {
      t: dq.regularHistory ? `Ventana temporal: ${dq.sampleCount}/240 muestras continuas` : `Ventana incompleta: ${dq.sampleCount}/240 muestras`,
      state: dq.regularHistory ? "ok" : "pending",
    },
    {
      t: dq.timestampMode === "source"
        ? "Timestamp proporcionado por OPC-UA"
        : dq.timestampMode === "receive"
          ? "Solo hora de recepción; origen no verificable"
          : dq.timestampMode === "simulated" ? "Datos del simulador" : "Sin datos disponibles",
      state: dq.timestampMode === "source" ? "ok" : dq.timestampMode === "simulated" ? "sim" : "pending",
    },
    { t: "Historiador PI/OSI: integración no verificada", state: "pending" },
  ];

  return (
    <div className="panel corners panel-glow p-3.5 flex-1 min-w-[280px] rise" style={{ animationDelay: "300ms" }}>
      <SectionHead level="QA" title="¿Los datos sirven?" accent="#7fb4e6" />
      <div className="flex items-start gap-3.5">
        <div className="shrink-0 text-center">
          <div className="font-mono text-[30px] font-semibold leading-none tabular-nums" style={{ color: gradeColor }}>
            {dq.overall}
          </div>
          <div
            className="mt-1.5 inline-block font-mono text-[11px] font-bold px-2 py-[1px] border"
            style={{ color: gradeColor, borderColor: `${gradeColor}66`, background: `${gradeColor}10` }}
          >
            {dq.grade}
          </div>
          <div className="label-mono mt-1.5 !text-[8px]">score /100</div>
        </div>

        <div className="flex-1 min-w-0">
          <div className="space-y-1">
            {dq.rows.map((r) => {
              const worst = Math.max(r.flat * 2, r.spikes * 6);
              const c = worst > 30 ? "#f26d5f" : worst > 10 ? "#f2b33d" : "#3fd0b6";
              return (
                <div key={r.key} className="flex items-center gap-2">
                  <span className="font-mono text-[9px] text-fg3 w-12 shrink-0">{VAR_META[r.key].tag}</span>
                  <span className="flex-1 h-[3px] bg-[#1b2e37] overflow-hidden">
                    <span className="block h-full ease-bar" style={{ width: `${Math.min(100, 100 - worst)}%`, background: c }} />
                  </span>
                  <span className="font-mono text-[8.5px] text-fg3 w-[74px] text-right shrink-0 tabular-nums">
                    plana {r.flat.toFixed(0)}% · {r.spikes} spikes
                  </span>
                </div>
              );
            })}
          </div>
          <p className="text-[10.5px] text-fg2 mt-2 leading-snug">{dq.verdict}</p>
        </div>
      </div>

      <div className="mt-3 pt-2.5 border-t border-line grid grid-cols-1 sm:grid-cols-3 gap-1.5">
        {checklist.map((c, i) => (
          <div key={i} className="flex items-center gap-1.5 text-[9.5px]">
            {c.state === "ok" ? (
              <IconCheck size={11} className="text-ok shrink-0" />
            ) : c.state === "pending" ? (
              <span className="w-[11px] h-[11px] grid place-items-center shrink-0 font-mono text-[9px] text-watch border border-[#f2b33d55]">?</span>
            ) : (
              <IconBolt size={11} className="text-fc shrink-0" />
            )}
            <span className={c.state === "pending" ? "text-watch" : "text-fg3"}>{c.t}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[9.5px] text-fg3 leading-snug">
        El clasificador (N3) entrena con ventanas etiquetadas del simulador: para un modelo
        supervisado “falla en X horas” con datos reales se requieren históricos de intervenciones
        documentadas.
      </p>
    </div>
  );
}

// ------------------------- controles de simulación -------------------------
const SCENARIO_BTNS: { s: Scenario; icon: typeof IconDroplet }[] = [
  { s: "liquidLoading", icon: IconDroplet },
  { s: "restriction", icon: IconValve },
  { s: "sensorFault", icon: IconChip },
  { s: "controlIssue", icon: IconGauge },
  { s: "casingLeak", icon: IconValve },
  { s: "tubingLeak", icon: IconValve },
  { s: "hydrates", icon: IconDroplet },
  { s: "sanding", icon: IconGauge },
];

export function ScenarioControls({ onInject, active }: { onInject: (s: Scenario) => void; active: Scenario }) {
  return (
    <div className="panel p-3.5 rise" style={{ animationDelay: "360ms" }}>
      <SectionHead level="SIM" title="Inyectar régimen" accent="#f2873d" />
      <div className="flex flex-wrap gap-1.5">
        {SCENARIO_BTNS.map(({ s, icon: Ic }) => {
          const isActive = s === active;
          return (
            <button
              key={s}
              onClick={() => onInject(s)}
              className={`inline-flex items-center gap-1.5 font-mono text-[10px] tracking-[0.06em] px-2.5 py-[7px] border transition-all duration-150 ${
                isActive
                  ? "text-alert border-[#f2873d77] bg-[#f2873d14] font-semibold"
                  : "text-fg2 border-line hover:border-line2 hover:text-fg hover:-translate-y-[1px]"
              }`}
            >
              <Ic size={12} />
              {SCENARIO_INFO[s].label}
            </button>
          );
        })}
        <button
          onClick={() => onInject("normal")}
          className={`inline-flex items-center gap-1.5 font-mono text-[10px] tracking-[0.06em] px-2.5 py-[7px] border transition-all duration-150 ${
            active === "normal"
              ? "text-ok border-[#3fd0b677] bg-[#3fd0b612] font-semibold"
              : "text-fg2 border-line hover:border-[#3fd0b655] hover:text-ok hover:-translate-y-[1px]"
          }`}
        >
          <IconCheck size={12} />
          Estabilizar pozo
        </button>
      </div>
      <p className="mt-2 text-[9.5px] text-fg3 leading-snug">
        Demo: inyecta un régimen físico en la telemetría sintética y observa cómo reacciona el pipeline N1 → N5 en vivo.
      </p>
    </div>
  );
}
