// ---------------------------------------------------------------------------
// VIGÍA ML · motor de inferencia y entrenamiento (TensorFlow.js)
// N1 LSTM de pronóstico multivariado (rollout recursivo, paso 15 min)
// N2 Autoencoder denso de detección de anomalías (error de reconstrucción)
// N3 Clasificador denso softmax de diagnóstico (5 clases operativas)
//
// Todo entrena EN EL NAVEGADOR sobre telemetría sintética del simulador.
// El bucle de entrenamiento trabaja POR LOTES y cede el hilo principal
// (tf.nextFrame) entre lote y lote: la interfaz nunca se congela aunque el
// navegador caiga al backend CPU. Incluye sonda de WebGL, presupuesto de
// tiempo por fase, carga adaptativa (mitad de pozos y épocas en CPU) y
// cancelación en caliente.
// ---------------------------------------------------------------------------
import * as tf from "@tensorflow/tfjs";
import {
  AE_WIN,
  CLASSES,
  CLS_WIN,
  FC_INPUT,
  FC_MIN,
  FC_STEPS,
  ML_VARS,
  aggregates,
  extractFeatures,
  generateDataset,
  makeNorm,
  normVec,
} from "./dataGen";
import type { ClassId } from "./dataGen";
import { anomalyLevel, linSlopePerHour } from "../models";
import type { Contribution, Forecast, Hypothesis } from "../models";
import { hasRegularMinuteCadence } from "../sim";
import type { Sample, VarKey } from "../sim";

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const clamp01 = (v: number) => clamp(v, 0, 1);
const ABORT_MSG = "__vigia_abort__";

// ------------------------------ estado público ------------------------------
export type MlPhase =
  | "idle"
  | "data"
  | "forecast"
  | "autoencoder"
  | "classifier"
  | "ready"
  | "error";

export interface ModelCard {
  name: string;
  kind: string;
  params: number;
  valLoss: number;
  acc?: number;
  epochs: number;
  trainMs: number;
}

export interface TrainState {
  phase: MlPhase;
  label: string;
  progress: number; // 0..1 global
  epoch: number;
  epochs: number;
  loss: number;
  valLoss: number;
  acc: number;
  lossHist: number[];
  accHist: number[];
  confusion: number[][] | null;
  dataset: { wells: number; fc: number; ae: number; cls: number };
  cards: ModelCard[];
  backend: string;
  elapsedMs: number;
  error: string | null;
}

export interface MlAnomaly {
  score: number;
  level: ReturnType<typeof anomalyLevel>;
  contributions: Contribution[];
  reconErr: number;
}

export interface ClassProb {
  cls: ClassId;
  id: string; // id de hipótesis compatible con N5
  p: number;
}

export interface MlForecast {
  stepMin: number;
  byVar: Record<VarKey, { mean: number[]; lo: number[]; hi: number[] }>;
  sd: Record<VarKey, number>;
}

const HYP_BY_CLASS: Record<ClassId, { id: string; name: string; icon: Hypothesis["icon"] }> = {
  normal: { id: "normal", name: "Operación dentro de envolvente", icon: "check" },
  liquidLoading: { id: "liquid-loading", name: "Liquid loading en tubing", icon: "droplet" },
  restriction: { id: "restriction", name: "Restricción en línea (hidratos / escala)", icon: "valve" },
  sensorFault: { id: "sensor-pt", name: "Sensor defectuoso · PT-101", icon: "chip" },
  controlIssue: { id: "control-issue", name: "Problema de control · actuador de choke", icon: "gauge" },
};

// ------------------------------ el motor ------------------------------------
export class VigiaEngine {
  ready = false;
  backend = "—";
  private aborted = false;
  private state: TrainState = VigiaEngine.initialState();

  private fc: tf.LayersModel | null = null; // N1 LSTM
  private ae: tf.LayersModel | null = null; // N2 autoencoder
  private cls: tf.LayersModel | null = null; // N3 clasificador
  private fcResid: number[] = []; // σ de residuales por salida [36]
  private aeCal = { mu: 0, sigma: 1, perVar: [1, 1, 1, 1, 1, 1] as number[] };

  static initialState(): TrainState {
    return {
      phase: "idle",
      label: "Inicializando motor…",
      progress: 0,
      epoch: 0,
      epochs: 0,
      loss: 0,
      valLoss: 0,
      acc: 0,
      lossHist: [],
      accHist: [],
      confusion: null,
      dataset: { wells: 0, fc: 0, ae: 0, cls: 0 },
      cards: [],
      backend: "—",
      elapsedMs: 0,
      error: null,
    };
  }

  get currentState(): TrainState {
    return this.state;
  }

  dispose() {
    this.ready = false;
    this.fc?.dispose();
    this.ae?.dispose();
    this.cls?.dispose();
    this.fc = null;
    this.ae = null;
    this.cls = null;
  }

  /** Cancela el entrenamiento en curso (el bucle termina en el próximo lote). */
  abort() {
    this.aborted = true;
    this.dispose();
  }

  // ------------------------------ arquitecturas -----------------------------

  private buildForecaster(): tf.LayersModel {
    const m = tf.sequential();
    m.add(
      tf.layers.lstm({
        units: 28,
        inputShape: [FC_INPUT, ML_VARS.length],
        returnSequences: false,
      }),
    );
    m.add(tf.layers.dropout({ rate: 0.08 }));
    m.add(tf.layers.dense({ units: FC_STEPS * ML_VARS.length })); // 36 salidas
    m.compile({ optimizer: tf.train.adam(0.008), loss: "meanSquaredError" });
    return m;
  }

  private buildAutoencoder(): tf.LayersModel {
    const m = tf.sequential();
    const flat = AE_WIN * ML_VARS.length; // 180
    m.add(tf.layers.dense({ units: 72, activation: "relu", inputShape: [flat] }));
    m.add(tf.layers.dense({ units: 20, activation: "relu" })); // código latente
    m.add(tf.layers.dense({ units: 72, activation: "relu" }));
    m.add(tf.layers.dense({ units: flat }));
    m.compile({ optimizer: tf.train.adam(0.006), loss: "meanSquaredError" });
    return m;
  }

  private buildClassifier(nFeatures: number): tf.LayersModel {
    const m = tf.sequential();
    m.add(tf.layers.dense({ units: 32, activation: "relu", inputShape: [nFeatures] }));
    m.add(tf.layers.dropout({ rate: 0.1 }));
    m.add(tf.layers.dense({ units: 16, activation: "relu" }));
    m.add(tf.layers.dense({ units: CLASSES.length, activation: "softmax" }));
    m.compile({
      optimizer: tf.train.adam(0.004),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
    return m;
  }

  // ------------------------------ entrenamiento -----------------------------

  /** Predicción por bloques cediendo el hilo: evita bloqueos con sets grandes. */
  private async predictRows(
    model: tf.LayersModel,
    x: tf.Tensor,
    rows: number,
    chunk = 64,
  ): Promise<number[]> {
    const out: number[] = [];
    for (let i = 0; i < rows; i += chunk) {
      if (this.aborted) throw new Error(ABORT_MSG);
      const k = Math.min(chunk, rows - i);
      const xs = x.slice([i], [k]);
      const p = model.predict(xs) as tf.Tensor;
      out.push(...Array.from(await p.data()));
      xs.dispose();
      p.dispose();
      if (i + chunk < rows) await tf.nextFrame();
    }
    return out;
  }

  /**
   * Bucle de entrenamiento por lotes. A diferencia de model.fit(), aquí se
   * cede el hilo principal entre lote y lote (tf.nextFrame), así que la
   * interfaz respira durante TODO el entrenamiento.
   * Devuelve el nº de épocas completadas (puede ser < epochs por presupuesto).
   */
  private async fitLoop(o: {
    model: tf.LayersModel;
    n: number;
    makeBatch: (rows: number[]) => [tf.Tensor, tf.Tensor];
    valX: tf.Tensor;
    valY: tf.Tensor;
    epochs: number;
    batchSize: number;
    yieldEvery: number;
    budgetMs: number;
    withAcc: boolean;
    onEpoch: (ep: number, logs: { loss: number; valLoss: number; acc: number }) => void;
    onBatch?: (frac: number, loss: number) => void;
  }): Promise<number> {
    const n = o.n;
    if (n === 0) return 0;
    let rs = (0x9e3779b9 ^ (o.epochs * 2654435761)) >>> 0;
    const rnd = () => {
      rs = (rs * 1664525 + 1013904223) >>> 0;
      return rs / 4294967296;
    };
    const idx = Array.from({ length: n }, (_, i) => i);
    const t0 = performance.now();
    let done = 0;
    for (let ep = 0; ep < o.epochs; ep++) {
      if (this.aborted) throw new Error(ABORT_MSG);
      // barajado determinista por época
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rnd() * (i + 1));
        [idx[i], idx[j]] = [idx[j], idx[i]];
      }
      let lossSum = 0;
      let nb = 0;
      for (let start = 0; start < n; start += o.batchSize) {
        if (this.aborted) throw new Error(ABORT_MSG);
        const rows = idx.slice(start, start + o.batchSize);
        const [tx, ty] = o.makeBatch(rows);
        const out = (await o.model.trainOnBatch(tx, ty)) as number | number[];
        tx.dispose();
        ty.dispose();
        const l = Array.isArray(out) ? out[0] : out;
        lossSum += l;
        nb++;
        o.onBatch?.((ep + (start + o.batchSize) / n) / o.epochs, l);
        if (nb % o.yieldEvery === 0) await tf.nextFrame();
      }
      // validación de la época (subconjunto fijo elegido por el llamador)
      const ev = o.model.evaluate(o.valX, o.valY, { batchSize: 128 }) as tf.Scalar | tf.Scalar[];
      const sc = Array.isArray(ev) ? ev : [ev];
      const vals = await Promise.all(sc.map((x) => x.data()));
      sc.forEach((x) => x.dispose());
      const valLoss = vals[0] ? (vals[0][0] ?? 0) : 0;
      const valAcc = o.withAcc && vals.length > 1 && vals[1] ? (vals[1][0] ?? 0) : 0;
      o.onEpoch(ep, {
        loss: lossSum / Math.max(1, nb),
        valLoss,
        acc: valAcc,
      });
      await tf.nextFrame();
      done = ep + 1;
      if (performance.now() - t0 > o.budgetMs) break; // presupuesto agotado
    }
    return done;
  }

  async trainAll(cb: (s: TrainState) => void): Promise<boolean> {
    const t0 = performance.now();
    this.ready = false;
    this.aborted = false;
    this.dispose();
    const st = VigiaEngine.initialState();
    let lastSoft = performance.now();
    const emit = (p: Partial<TrainState>) => {
      Object.assign(st, p, { elapsedMs: Math.round(performance.now() - t0) });
      this.state = { ...st };
      cb(this.state);
    };
    const emitSoft = (p: Partial<TrainState>) => {
      const now = performance.now();
      if (now - lastSoft < 160) return; // máx. ~6 actualizaciones/s de progreso
      lastSoft = now;
      emit(p);
    };

    try {
      emit({ phase: "data", label: "Inicializando TensorFlow.js…", progress: 0.02 });
      await tf.ready();
      await tf.nextFrame();

      // sonda del backend: si WebGL está declarado pero no computa, caer a CPU
      if (tf.getBackend() === "webgl") {
        try {
          const probe = tf.tidy(() => tf.matMul(tf.ones([96, 96]), tf.ones([96, 96])));
          await probe.data();
          probe.dispose();
        } catch {
          try {
            await tf.setBackend("cpu");
            await tf.ready();
          } catch {
            /* se mantiene el backend actual */
          }
        }
      }
      this.backend = tf.getBackend();
      const cpu = this.backend !== "webgl";
      const budgetMs = cpu ? 80_000 : 120_000;
      const E1 = cpu ? 13 : 22;
      const E2 = cpu ? 11 : 20;
      const E3 = cpu ? 17 : 30;
      const yieldEvery = cpu ? 1 : 2;
      emit({
        backend: this.backend,
        phase: "data",
        label: `Backend ${this.backend.toUpperCase()} · generando telemetría sintética…`,
        progress: 0.03,
      });
      await tf.nextFrame();

      const ds = await generateDataset({
        wellsScale: cpu ? 0.7 : 1,
        onProgress: async (frac) => {
          emitSoft({
            phase: "data",
            label: `Generando telemetría sintética… ${Math.round(frac * 100)}%`,
            progress: 0.02 + 0.04 * frac,
          });
          await tf.nextFrame();
        },
      });
      const nFc = ds.fcTrainX.length + ds.fcValX.length;
      const nAe = ds.aeTrainX.length + ds.aeValX.length;
      const nCls = ds.clsTrainX.length + ds.clsValX.length;
      emit({
        phase: "data",
        label: `Dataset listo · ${ds.wells} pozos sintéticos`,
        progress: 0.06,
        dataset: { wells: ds.wells, fc: nFc, ae: nAe, cls: nCls },
      });
      await tf.nextFrame();

      // ------------------------- N1 · LSTM pronóstico -----------------------
      let tPh = performance.now();
      emit({ phase: "forecast", label: "Entrenando LSTM de pronóstico (N1)…", epochs: E1, epoch: 0, lossHist: [] });
      this.fc = this.buildForecaster();
      const v1 = Math.min(160, ds.fcValX.length);
      const fcVX = tf.tensor3d(ds.fcValX.slice(0, v1));
      const fcVY = tf.tensor2d(ds.fcValY.slice(0, v1));
      let fcValLoss = 0;
      const done1 = await this.fitLoop({
        model: this.fc,
        n: ds.fcTrainX.length,
        makeBatch: (rows) => [
          tf.tensor3d(rows.map((i) => ds.fcTrainX[i])),
          tf.tensor2d(rows.map((i) => ds.fcTrainY[i])),
        ],
        valX: fcVX,
        valY: fcVY,
        epochs: E1,
        batchSize: 32,
        yieldEvery,
        budgetMs: budgetMs * 0.5,
        withAcc: false,
        onEpoch: (ep, lg) => {
          fcValLoss = lg.valLoss || fcValLoss;
          emit({
            phase: "forecast",
            epoch: ep + 1,
            loss: lg.loss,
            valLoss: lg.valLoss,
            lossHist: [...st.lossHist.slice(-79), lg.loss],
            progress: 0.06 + 0.44 * ((ep + 1) / E1),
          });
        },
        onBatch: (frac, l) => emitSoft({ phase: "forecast", loss: l, progress: 0.06 + 0.44 * frac }),
      });
      fcVX.dispose();
      fcVY.dispose();

      // calibración: σ de residuales por (paso, variable) sobre validación
      {
        const xAll = tf.tensor3d(ds.fcValX);
        const arr = await this.predictRows(this.fc, xAll, ds.fcValX.length);
        xAll.dispose();
        const n = ds.fcValX.length;
        const D = FC_STEPS * ML_VARS.length;
        const stds: number[] = [];
        for (let c = 0; c < D; c++) {
          let s = 0, s2 = 0;
          for (let i = 0; i < n; i++) {
            const e = arr[i * D + c] - ds.fcValY[i][c];
            s += e;
            s2 += e * e;
          }
          const mu = s / n;
          stds.push(Math.sqrt(Math.max(1e-8, s2 / n - mu * mu)));
        }
        this.fcResid = stds;
      }
      emit({
        progress: 0.5,
        cards: [
          ...st.cards,
          {
            name: "LSTM pronóstico",
            kind: `LSTM(28) → Dense(36) · entrada ${FC_INPUT}×6 · rollout ${FC_STEPS} pasos${done1 < E1 ? " · recortado por presupuesto de tiempo" : ""}`,
            params: this.fc.countParams(),
            valLoss: fcValLoss,
            epochs: done1,
            trainMs: Math.round(performance.now() - tPh),
          },
        ],
      });
      await tf.nextFrame();

      // ---------------------- N2 · Autoencoder anomalías --------------------
      tPh = performance.now();
      emit({
        phase: "autoencoder",
        label: "Entrenando autoencoder de anomalías (N2)…",
        epochs: E2,
        epoch: 0,
        lossHist: [],
      });
      this.ae = this.buildAutoencoder();
      const flatRow = (w: number[][]) => w.flat();
      const v2 = Math.min(160, ds.aeValX.length);
      const aeVX = tf.tensor2d(ds.aeValX.slice(0, v2).map(flatRow));
      let aeValLoss = 0;
      const done2 = await this.fitLoop({
        model: this.ae,
        n: ds.aeTrainX.length,
        makeBatch: (rows) => {
          const bx = tf.tensor2d(rows.map((i) => flatRow(ds.aeTrainX[i])));
          return [bx, bx]; // el autoencoder reconstruye su propia entrada
        },
        valX: aeVX,
        valY: aeVX,
        epochs: E2,
        batchSize: 32,
        yieldEvery,
        budgetMs: budgetMs * 0.25,
        withAcc: false,
        onEpoch: (ep, lg) => {
          aeValLoss = lg.valLoss || aeValLoss;
          emit({
            phase: "autoencoder",
            epoch: ep + 1,
            loss: lg.loss,
            valLoss: lg.valLoss,
            lossHist: [...st.lossHist.slice(-79), lg.loss],
            progress: 0.5 + 0.22 * ((ep + 1) / E2),
          });
        },
        onBatch: (frac, l) => emitSoft({ phase: "autoencoder", loss: l, progress: 0.5 + 0.22 * frac }),
      });
      aeVX.dispose();

      // calibración del umbral de error sobre operación normal
      {
        const flatVal = ds.aeValX.map(flatRow);
        const xAll = tf.tensor2d(flatVal);
        const arr = await this.predictRows(this.ae, xAll, flatVal.length);
        xAll.dispose();
        const n = flatVal.length;
        const errs: number[] = [];
        const perVar = new Array(ML_VARS.length).fill(0);
        for (let i = 0; i < n; i++) {
          let e = 0;
          for (let t = 0; t < AE_WIN; t++) {
            for (let k = 0; k < ML_VARS.length; k++) {
              const d = arr[i * AE_WIN * ML_VARS.length + t * ML_VARS.length + k] - flatVal[i][t * ML_VARS.length + k];
              e += (d * d) / (AE_WIN * ML_VARS.length);
              perVar[k] += (d * d) / (AE_WIN * n);
            }
          }
          errs.push(e);
        }
        const mu = errs.reduce((a, b) => a + b, 0) / n;
        const sigma = Math.sqrt(errs.reduce((a, b) => a + (b - mu) ** 2, 0) / n);
        this.aeCal = { mu, sigma: Math.max(1e-9, sigma), perVar: perVar.map((v) => Math.max(1e-9, v)) };
      }
      emit({
        progress: 0.72,
        cards: [
          ...st.cards,
          {
            name: "Autoencoder N2",
            kind: `Dense 180→72→20→72→180 · ventana ${AE_WIN} min · σ calibrada en validación${done2 < E2 ? " · recortado" : ""}`,
            params: this.ae.countParams(),
            valLoss: aeValLoss,
            epochs: done2,
            trainMs: Math.round(performance.now() - tPh),
          },
        ],
      });
      await tf.nextFrame();

      // --------------------- N3 · Clasificador diagnóstico -------------------
      tPh = performance.now();
      emit({
        phase: "classifier",
        label: "Entrenando clasificador de diagnóstico (N3)…",
        epochs: E3,
        epoch: 0,
        lossHist: [],
        accHist: [],
      });
      this.cls = this.buildClassifier(ds.clsTrainX[0]?.length ?? 18);
      const nFeats = ds.clsTrainX[0]?.length ?? 18;
      const oneHot = (y: number) =>
        Array.from({ length: CLASSES.length }, (_, i) => (i === y ? 1 : 0));
      const v3 = Math.min(160, ds.clsValX.length);
      const clsVX = tf.tensor2d(ds.clsValX.slice(0, v3));
      const clsVY = tf.tensor2d(ds.clsValY.slice(0, v3).map(oneHot));
      let clsValLoss = 0;
      const done3 = await this.fitLoop({
        model: this.cls,
        n: ds.clsTrainX.length,
        makeBatch: (rows) => [
          tf.tensor2d(rows.map((i) => ds.clsTrainX[i])),
          tf.tensor2d(rows.map((i) => oneHot(ds.clsTrainY[i]))),
        ],
        valX: clsVX,
        valY: clsVY,
        epochs: E3,
        batchSize: 32,
        yieldEvery,
        budgetMs: budgetMs, // todo el tiempo restante del presupuesto global
        withAcc: true,
        onEpoch: (ep, lg) => {
          clsValLoss = lg.valLoss || clsValLoss;
          emit({
            phase: "classifier",
            epoch: ep + 1,
            loss: lg.loss,
            valLoss: lg.valLoss,
            acc: lg.acc,
            lossHist: [...st.lossHist.slice(-79), lg.loss],
            accHist: [...st.accHist.slice(-79), lg.acc],
            progress: 0.72 + 0.27 * ((ep + 1) / E3),
          });
        },
        onBatch: (frac, l) => emitSoft({ phase: "classifier", loss: l, progress: 0.72 + 0.27 * frac }),
      });
      clsVX.dispose();
      clsVY.dispose();

      // matriz de confusión sobre validación completa
      let confusion: number[][] = [];
      let valAcc = 0;
      {
        const xAll = tf.tensor2d(ds.clsValX);
        const arr = await this.predictRows(this.cls, xAll, ds.clsValX.length);
        xAll.dispose();
        confusion = Array.from({ length: CLASSES.length }, () =>
          Array.from({ length: CLASSES.length }, () => 0),
        );
        ds.clsValY.forEach((yTrue, i) => {
          let best = 0;
          for (let c = 1; c < CLASSES.length; c++) {
            if (arr[i * CLASSES.length + c] > arr[i * CLASSES.length + best]) best = c;
          }
          confusion[yTrue][best]++;
        });
        const total = ds.clsValY.length || 1;
        valAcc = confusion.reduce((a, row, i) => a + row[i], 0) / total;
      }
      emit({
        progress: 1,
        confusion,
        cards: [
          ...st.cards,
          {
            name: "Clasificador N3",
            kind: `Dense ${nFeats}→32→16→5 softmax · ${(valAcc * 100).toFixed(1)}% acc. validación${done3 < E3 ? " · recortado" : ""}`,
            params: this.cls.countParams(),
            valLoss: clsValLoss,
            acc: valAcc,
            epochs: done3,
            trainMs: Math.round(performance.now() - tPh),
          },
        ],
      });
      await tf.nextFrame();

      this.ready = true;
      emit({
        phase: "ready",
        label: `Modelos listos · backend ${this.backend.toUpperCase()} · inferencia en vivo`,
        progress: 1,
        epoch: 0,
        epochs: 0,
      });
      return true;
    } catch (err) {
      if (this.aborted || (err instanceof Error && err.message === ABORT_MSG)) {
        this.dispose();
        emit({
          phase: "idle",
          label: "Entrenamiento cancelado · el pipeline estadístico sigue operativo",
          progress: 0,
          epoch: 0,
          epochs: 0,
          error: null,
        });
        return false;
      }
      emit({
        phase: "error",
        label: "Error de entrenamiento",
        error: err instanceof Error ? err.message : String(err),
      });
      return false;
    }
  }

  // ------------------------------ inferencia --------------------------------

  /** N2 · score de anomalía 0-100 por error de reconstrucción + detector de señal plana. */
  async assess(samples: Sample[], base: Record<VarKey, number>): Promise<{ anom: MlAnomaly; probs: ClassProb[] } | null> {
    if (!this.ready || !this.ae || !this.cls) return null;
    const win = samples.slice(-AE_WIN);
    if (win.length < AE_WIN || !hasRegularMinuteCadence(win, AE_WIN)) return null;
    const norm = makeNorm(base);
    const flat = win.flatMap((s) => normVec(s, norm));

    // --- autoencoder ---
    const x = tf.tensor2d([flat]);
    const y = this.ae.predict(x) as tf.Tensor;
    const pred = Array.from(await y.data());
    x.dispose();
    y.dispose();

    const perVar = new Array(ML_VARS.length).fill(0);
    for (let t = 0; t < AE_WIN; t++) {
      for (let k = 0; k < ML_VARS.length; k++) {
        const d = pred[t * ML_VARS.length + k] - flat[t * ML_VARS.length + k];
        perVar[k] += (d * d) / AE_WIN;
      }
    }
    const totalErr = perVar.reduce((a, b) => a + b, 0) / ML_VARS.length;
    const z = (totalErr - this.aeCal.mu) / this.aeCal.sigma;

    // detector de señal congelada (un AE no ve "anómalo" un valor constante)
    const flatKeys: VarKey[] = [];
    for (const k of ["pt", "pc", "pl", "temp"] as VarKey[]) {
      const tail = win.slice(-25).map((s) => s[k]);
      const m = tail.reduce((a, b) => a + b, 0) / tail.length;
      const sd = Math.sqrt(tail.reduce((a, b) => a + (b - m) ** 2, 0) / tail.length);
      if (sd < 0.05) flatKeys.push(k);
    }

    let score = 100 * (1 - Math.exp(-Math.max(0, z - 1.2) / 5.2));
    if (flatKeys.length > 0) score = Math.max(score, 74);
    score = clamp(Math.round(score), 2, 98);

    const contributions: Contribution[] = [];
    ML_VARS.forEach((k, i) => {
      const ratio = perVar[i] / this.aeCal.perVar[i];
      const pz = clamp((ratio - 1) * 1.9 + 0.8, 0, 9);
      if (pz > 1.6) contributions.push({ key: k, z: pz });
    });
    for (const k of flatKeys) contributions.push({ key: k, z: 5.4 });
    contributions.sort((a, b) => b.z - a.z);

    // --- clasificador ---
    const win90 = samples.slice(-CLS_WIN);
    let probs: ClassProb[] = [];
    if (win90.length >= CLS_WIN && hasRegularMinuteCadence(win90, CLS_WIN)) {
      const f = extractFeatures(win90, norm);
      const fx = tf.tensor2d([f]);
      const fy = this.cls.predict(fx) as tf.Tensor;
      const p = Array.from(await fy.data());
      fx.dispose();
      fy.dispose();
      probs = CLASSES.map((c, i) => ({ cls: c, id: HYP_BY_CLASS[c].id, p: p[i] })).sort(
        (a, b) => b.p - a.p,
      );
    }

    return {
      anom: { score, level: anomalyLevel(score), contributions: contributions.slice(0, 3), reconErr: totalErr },
      probs,
    };
  }

  /** N1 · pronóstico multivariado con rollout recursivo (paso 15 min). */
  async forecastSeries(
    samples: Sample[],
    base: Record<VarKey, number>,
    steps: number,
  ): Promise<MlForecast | null> {
    if (!this.ready || !this.fc) return null;
    const norm = makeNorm(base);
    const need = FC_INPUT * FC_MIN;
    const buf = samples.slice(-need);
    if (buf.length < need || !hasRegularMinuteCadence(buf, need)) return null;

    let seq = aggregates(buf, norm); // [16][6] normalizado
    const blocks = Math.ceil(steps / FC_STEPS);
    for (let b = 0; b < blocks; b++) {
      const x = tf.tensor3d([seq.slice(-FC_INPUT)]);
      const y = this.fc.predict(x) as tf.Tensor;
      const out = Array.from(await y.data());
      x.dispose();
      y.dispose();
      for (let s = 0; s < FC_STEPS; s++) {
        seq = [...seq, out.slice(s * ML_VARS.length, (s + 1) * ML_VARS.length)];
      }
      if (b < blocks - 1) await tf.nextFrame(); // ceder el hilo entre bloques
    }

    const byVar = {} as MlForecast["byVar"];
    const sdVar = {} as MlForecast["sd"];
    ML_VARS.forEach((k, ki) => {
      const mean: number[] = [], lo: number[] = [], hi: number[] = [];
      let sdSum = 0;
      for (let g = 0; g < steps; g++) {
        const val = norm.base[k] + seq[FC_INPUT + g][ki] * norm.scale[k];
        const clamped =
          k === "q" ? Math.max(0, val) : k === "choke" ? clamp(val, 2, 100) : val;
        const block = Math.floor(g / FC_STEPS);
        const sd =
          this.fcResid[(g % FC_STEPS) * ML_VARS.length + ki] * Math.sqrt(block + 1);
        mean.push(clamped);
        lo.push(clamped - 1.28 * sd);
        hi.push(clamped + 1.28 * sd);
        sdSum += sd;
      }
      byVar[k] = { mean, lo, hi };
      sdVar[k] = sdSum / steps;
    });

    return { stepMin: FC_MIN, byVar, sd: sdVar };
  }
}

// --------------------- fusión ML + reglas para N3/N5 -------------------------
const f1 = (v: number) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}`;

function genericEvidence(cls: ClassId, win: Sample[]): string[] {
  const slope = (k: VarKey) => linSlopePerHour(win.map((s) => ({ x: s.m, y: s[k] })));
  const ptT = slope("pt"), pcT = slope("pc"), plT = slope("pl"), qT = slope("q");
  switch (cls) {
    case "liquidLoading":
      return [
        `P tubing ${f1(ptT)} psi/h · P casing ${f1(pcT)} psi/h`,
        `Caudal ${f1(qT)} Mscf/d por hora`,
        "Patrón multivariado clasificado como liquid loading por la red neuronal",
      ];
    case "restriction":
      return [
        `P línea ${f1(plT)} psi/h · P tubing ${f1(ptT)} psi/h`,
        `Caudal ${f1(qT)} Mscf/d por hora`,
        "Patrón clasificado como restricción en línea por la red neuronal",
      ];
    case "sensorFault":
      return [
        `P tubing ${f1(ptT)} psi/h con varianza deprimida`,
        "Señal compatible con transmisor congelado (clasificador ML)",
      ];
    case "controlIssue":
      return [
        `Choke en movimiento sin respuesta del caudal (${f1(qT)} Mscf/d por hora)`,
        "Patrón clasificado como problema de actuador por la red neuronal",
      ];
    default:
      return [
        "Todas las variables dentro de ±2σ de su línea base móvil",
        "Sin cruce de umbrales proyectado en las próximas 24 h",
      ];
  }
}

/** Combina la salida del clasificador con las reglas físicas de N3. */
export function mergeDiagnosis(rules: Hypothesis[], probs: ClassProb[], win: Sample[]): Hypothesis[] {
  const byId = new Map(rules.map((h) => [h.id, h]));
  const out: Hypothesis[] = [];

  for (const pr of probs) {
    if (pr.p < 0.06) continue;
    if (pr.cls === "normal" && pr.p < 0.35) continue; // no mostrar "normal" débil
    const meta = HYP_BY_CLASS[pr.cls];
    const rule = byId.get(meta.id);
    out.push({
      id: meta.id,
      name: rule?.name ?? meta.name,
      icon: rule?.icon ?? meta.icon,
      conf: clamp01(pr.p * 0.97),
      evidence: rule && rule.evidence.length > 0 ? rule.evidence : genericEvidence(pr.cls, win),
    });
    byId.delete(meta.id);
  }
  // hipótesis de reglas sin respaldo del clasificador → secundarias
  for (const h of byId.values()) {
    out.push({ ...h, conf: clamp01(h.conf * 0.5) });
  }
  out.sort((a, b) => b.conf - a.conf);
  return out.slice(0, 4);
}

export type { Forecast };

// ---------------------------------------------------------------------------
// Interfaz estructural del motor: la implementación de referencia es
// VigiaEngine (hilo principal) y EngineProxy (Web Worker). El resto de la
// app solo depende de esta forma, nunca de la clase concreta.
// ---------------------------------------------------------------------------
export interface EngineLike {
  ready: boolean;
  backend: string;
  trainAll(cb: (s: TrainState) => void): Promise<boolean>;
  abort(): void;
  dispose(): void;
  assess(
    samples: Sample[],
    base: Record<VarKey, number>,
  ): Promise<{ anom: MlAnomaly; probs: ClassProb[] } | null>;
  forecastSeries(
    samples: Sample[],
    base: Record<VarKey, number>,
    steps: number,
  ): Promise<MlForecast | null>;
}
