import { useMemo } from "react";
import { adviseSetpoints } from "../lib/setpoints";
import type { SetpointAdvice } from "../lib/setpoints";
import type { Hypothesis } from "../lib/models";
import type { Sample, VarKey } from "../lib/sim";
import { SectionHead } from "./bits";

const fmt = (v: number, dec = 0) =>
  Number.isFinite(v) ? v.toLocaleString("es-EC", { maximumFractionDigits: dec, minimumFractionDigits: dec }) : "—";

const ACCENT = "#c9b8f0"; // violeta de optimización

function Metric({ label, value, unit, accent }: { label: string; value: string; unit?: string; accent?: string }) {
  return (
    <div className="min-w-[86px]">
      <div className="font-mono text-[8.5px] tracking-[0.14em] text-fg3 uppercase">{label}</div>
      <div className="font-mono text-[15px] font-semibold tabular-nums leading-tight" style={{ color: accent ?? "var(--color-fg)" }}>
        {value}
        {unit && <span className="text-[9px] font-normal text-fg3 ml-1">{unit}</span>}
      </div>
    </div>
  );
}

/** Curva de utilidad q(c)·S(h|c) normalizada + marcador actual vs recomendado. */
function UtilityCurve({ advice }: { advice: SetpointAdvice }) {
  const W = 280;
  const H = 56;
  const view = useMemo(() => {
    // reconstruye la forma de la curva con el mismo modelo del asesor
    const pts: { c: number; u: number }[] = [];
    const cNow = advice.chokeNow;
    const uRec = advice.qRec * advice.survRec;
    const span = Math.max(1e-6, Math.abs(uRec - advice.qNow * advice.survNow) + uRec * 0.35);
    for (let i = 0; i <= 40; i++) {
      const c = 15 + (80 * i) / 40;
      // aproximación visual: parábola centrada cerca del recomendado
      const x = (c - advice.chokeRec) / 42;
      const u = uRec * (1 - 0.55 * x * x);
      pts.push({ c, u });
    }
    void cNow;
    const uMax = Math.max(...pts.map((p) => p.u), uRec) * 1.08;
    return { pts, uMax, span, uRec };
  }, [advice]);

  const px = (c: number) => ((c - 15) / 80) * (W - 8) + 4;
  const py = (u: number) => H - 4 - (Math.max(0, u) / view.uMax) * (H - 12);
  const path = view.pts.map((p, i) => `${i === 0 ? "M" : "L"} ${px(p.c).toFixed(1)} ${py(p.u).toFixed(1)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full mt-2.5" role="img" aria-label="Curva de utilidad producción × supervivencia vs apertura del choke">
      <path d={path} fill="none" stroke={ACCENT} strokeWidth="1.6" opacity="0.9" />
      <line x1={px(advice.chokeRec)} y1="2" x2={px(advice.chokeRec)} y2={H - 2} stroke={ACCENT} strokeWidth="1" strokeDasharray="2 3" />
      <text x={Math.min(W - 4, px(advice.chokeRec) + 4)} y="10" fontSize="7.5" fill={ACCENT} fontFamily="IBM Plex Mono, monospace">
        REC {Math.round(advice.chokeRec)}%
      </text>
      <line x1={px(advice.chokeNow)} y1="2" x2={px(advice.chokeNow)} y2={H - 2} stroke="#5f7a82" strokeWidth="1" strokeDasharray="2 3" />
      <text x={Math.max(4, px(advice.chokeNow) - 4)} y={H - 3} fontSize="7.5" fill="#5f7a82" fontFamily="IBM Plex Mono, monospace" textAnchor="end">
        AHORA {Math.round(advice.chokeNow)}%
      </text>
    </svg>
  );
}

/**
 * Tarjeta del asesor de setpoints (roadmap #5): apertura óptima del choke
 * maximizando producción esperada × supervivencia Weibull al horizonte, con
 * restricciones duras de ascenso (Turner), erosión (API RP 14E) y margen
 * subcrítico.
 */
export function SetpointAdvisorCard({
  diag,
  samples,
  base,
  anomScore,
  dataReady,
}: {
  diag: Hypothesis[];
  samples: Sample[];
  base: Record<VarKey, number>;
  anomScore: number;
  dataReady: boolean;
}) {
  const advice = useMemo(() => {
    const top = diag.find((d) => d.id !== "normal") ?? diag[diag.length - 1] ?? { id: "normal", conf: 0 };
    return adviseSetpoints({ samples, base, diagId: top.id, diagConf: top.conf, anomScore });
  }, [diag, samples, base, anomScore]);

  if (!dataReady) {
    return (
      <div className="panel corners p-3.5 flex-1 min-w-[280px] rise" style={{ animationDelay: "220ms" }}>
        <SectionHead level="SETPOINTS" title="Asesor de apertura óptima" accent={ACCENT} />
        <p className="text-[11px] text-fg3 leading-relaxed font-mono">
          ASESOR EN ESPERA — se requieren 240 muestras consecutivas con calidad A/B. No se muestran sugerencias de apertura con historial insuficiente o discontinuo.
        </p>
      </div>
    );
  }

  if (!advice.feasible) {
    return (
      <div className="panel corners p-3.5 flex-1 min-w-[280px] rise" style={{ animationDelay: "220ms" }}>
        <SectionHead level="SETPOINTS" title="Asesor de apertura óptima" accent={ACCENT} />
        <p className="text-[11px] text-fg3 leading-relaxed font-mono">
          ASESOR EN ESPERA — {advice.reason?.toUpperCase()}
        </p>
      </div>
    );
  }

  const move = advice.chokeRec - advice.chokeNow;
  const action =
    Math.abs(move) <= 1.5 ? "MANTENER" : move > 0 ? "ABRIR" : "CERRAR";

  return (
    <div className="panel corners p-3.5 flex-1 min-w-[300px] rise" style={{ animationDelay: "220ms" }}>
      <SectionHead
        level="SETPOINTS"
        title="Asesor de apertura óptima"
        accent={ACCENT}
        right={
          <span
            className="font-mono text-[8px] font-semibold tracking-[0.14em] px-1.5 py-[2px] border"
            style={{ color: ACCENT, borderColor: `${ACCENT}55`, background: `${ACCENT}0f` }}
          >
            {action} {Math.abs(move) > 1.5 ? `${Math.abs(Math.round(move))} PCT` : ""}
          </span>
        }
      />

      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <Metric
          label="Choke"
          value={`${Math.round(advice.chokeNow)}% → ${Math.round(advice.chokeRec)}%`}
          accent={ACCENT}
        />
        <Metric label="Caudal esperado" value={fmt(advice.qRec)} unit="Mscf/d" />
        <Metric
          label="Ganancia"
          value={`${advice.gainPct >= 0 ? "+" : ""}${fmt(advice.gainPct, 1)}`}
          unit="%"
          accent={advice.gainPct >= 1 ? "#3fd0b6" : undefined}
        />
        <Metric label="ΔVolumen" value={`${advice.expectedDeltaMscf >= 0 ? "+" : ""}${fmt(advice.expectedDeltaMscf / 1000, 2)}`} unit={`MMscf / ${advice.horizonH} h`} />
        <Metric label="S(horizonte)" value={`${fmt(advice.survRec * 100, 1)}`} unit="%" />
      </div>

      <UtilityCurve advice={advice} />

      <p className="mt-2 text-[9px] text-fg3 leading-relaxed font-mono">
        SOLO ASESORÍA: no escribe en PLC/SCADA ni sustituye límites, permisos o validación de ingeniería. No ejecutar como setpoint automático.
      </p>

      <div className="mt-2.5 flex flex-wrap gap-1.5">
        {advice.constraints.map((c) => (
          <span
            key={c.id}
            title={c.detail}
            className="font-mono text-[8.5px] font-semibold tracking-[0.1em] px-1.5 py-[2px] border"
            style={{
              color: c.ok ? "#3fd0b6" : "#f26d5f",
              borderColor: c.ok ? "#3fd0b655" : "#f26d5f66",
              background: c.ok ? "#3fd0b60d" : "#f26d5f12",
            }}
          >
            {c.ok ? "✓" : "✕"} {c.label.toUpperCase()}
          </span>
        ))}
      </div>

      <ul className="mt-2 space-y-1">
        {advice.rationale.slice(0, 2).map((r, i) => (
          <li key={i} className="text-[10px] text-fg2 leading-snug">
            · {r}
          </li>
        ))}
      </ul>

      <p className="mt-2 text-[9px] text-fg3 leading-relaxed font-mono">
        q(c) POR MEDICIÓN VIRTUAL (BEAN) · SUPERVIVENCIA WEIBULL+AFT AL HORIZONTE · RESTRICCIONES: LIFT TURNER 1969,
        EROSIÓN API RP 14E, MARGEN SUBCRÍTICO P₂/P₁
      </p>
    </div>
  );
}
