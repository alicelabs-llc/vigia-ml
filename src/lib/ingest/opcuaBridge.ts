// ---------------------------------------------------------------------------
// VIGÍA · capa de ingesta de fuentes externas (v0.8.0)
// El navegador no habla el protocolo binario OPC-UA: los datos entran por un
// puente WebSocket (bridge/opcua-bridge.mjs en el repo). Este módulo define el
// contrato de las fuentes y el cliente del puente: reconexión con backoff,
// vigilancia de latidos y saneamiento de cada muestra antes de dejarla pasar.
// ---------------------------------------------------------------------------

import type { VarKey } from "../sim";

/** Estado del enlace con la fuente. */
export type LinkStatus = "idle" | "connecting" | "online" | "degraded" | "error";

/** Mapa variable → identificador de nodo en el servidor OPC-UA. */
export type TagMap = Record<VarKey, string>;

/** Muestra cruda de la fuente (solo variables presentes y finitas). */
export interface IngestSample {
  ts: number; // epoch ms usado para agrupar por minuto
  timestampOrigin?: "source" | "receive";
  values: Partial<Record<VarKey, number>>;
}

export interface LinkStats {
  received: number;
  accepted: number;
  rejected: number;
  reconnects: number;
}

export interface SourceAdapter {
  readonly id: string;
  readonly label: string;
  connect(url: string, tags: TagMap): void;
  disconnect(): void;
  readonly status: LinkStatus;
  readonly stats: LinkStats;
  onData(cb: (s: IngestSample) => void): void;
  onStatus(cb: (st: LinkStatus, info: string) => void): void;
}

/** Subconjunto de WebSocket que usa el cliente — inyectable en pruebas. */
export interface WSLike {
  send(data: string): void;
  close(code?: number, reason?: string): void;
  readyState: number;
  addEventListener(type: "open" | "message" | "close" | "error", cb: (ev: unknown) => void): void;
}

export interface BridgeOptions {
  makeWs?: (url: string) => WSLike; // fábrica inyectable (tests)
  reconnectBaseMs?: number; // def 1000
  reconnectMaxMs?: number; // def 15000
  heartbeatMs?: number; // sin mensajes ⇒ degraded (def 60000)
  heartbeatDeadMs?: number; // sin mensajes ⇒ corte forzado (def 120000)
}

type StatusCb = (st: LinkStatus, info: string) => void;
type DataCb = (s: IngestSample) => void;

interface Frame {
  op?: string;
  ts?: number;
  timestampOrigin?: "source" | "receive";
  values?: Record<string, unknown>;
  state?: string;
  msg?: string;
}

export class OpcuaBridgeSource implements SourceAdapter {
  readonly id = "opcua-bridge";
  readonly label = "Puente OPC-UA";

  private ws: WSLike | null = null;
  private url = "";
  private tags: TagMap | null = null;
  private closedByUs = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private hbTimer: ReturnType<typeof setInterval> | null = null;
  private lastMsgAt = 0;
  private attempts = 0;
  private statusValue: LinkStatus = "idle";
  private statsValue: LinkStats = { received: 0, accepted: 0, rejected: 0, reconnects: 0 };
  private dataCbs: DataCb[] = [];
  private statusCbs: StatusCb[] = [];
  private opts: Required<Pick<BridgeOptions, "reconnectBaseMs" | "reconnectMaxMs" | "heartbeatMs" | "heartbeatDeadMs">> &
    Pick<BridgeOptions, "makeWs">;
  private info = "";

  constructor(opts: BridgeOptions = {}) {
    this.opts = {
      reconnectBaseMs: opts.reconnectBaseMs ?? 1000,
      reconnectMaxMs: opts.reconnectMaxMs ?? 15000,
      heartbeatMs: opts.heartbeatMs ?? 60000,
      heartbeatDeadMs: opts.heartbeatDeadMs ?? 120000,
      ...(opts.makeWs ? { makeWs: opts.makeWs } : {}),
    };
  }

  get status(): LinkStatus {
    return this.statusValue;
  }
  get stats(): LinkStats {
    return { ...this.statsValue };
  }
  get lastInfo(): string {
    return this.info;
  }

  onData(cb: DataCb) {
    this.dataCbs.push(cb);
  }
  onStatus(cb: StatusCb) {
    this.statusCbs.push(cb);
  }

  private setStatus(st: LinkStatus, info: string) {
    this.statusValue = st;
    this.info = info;
    for (const cb of this.statusCbs) cb(st, info);
  }

  connect(url: string, tags: TagMap) {
    this.disconnect(); // limpiar cualquier enlace previo
    this.closedByUs = false;
    this.url = url;
    this.tags = { ...tags };
    this.open();
  }

  private makeSocket(): WSLike {
    if (this.opts.makeWs) return this.opts.makeWs(this.url);
    return new WebSocket(this.url) as unknown as WSLike;
  }

  private open() {
    if (!this.tags) return;
    this.setStatus("connecting", `conectando a ${this.url}`);
    let ws: WSLike;
    try {
      ws = this.makeSocket();
    } catch (e) {
      this.setStatus("error", `fallo al abrir socket: ${(e as Error)?.message ?? e}`);
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;

    ws.addEventListener("open", () => {
      // el mapa viaja en el subscribe: el puente traduce NodeId → etiqueta
      ws.send(JSON.stringify({ op: "subscribe", tags: this.tags, intervalMs: 15000 }));
      this.lastMsgAt = Date.now();
      this.startHeartbeat();
    });

    ws.addEventListener("message", (ev: unknown) => {
      this.lastMsgAt = Date.now();
      const raw = (ev as { data?: unknown })?.data;
      const text = typeof raw === "string" ? raw : "";
      let f: Frame;
      try {
        f = JSON.parse(text) as Frame;
      } catch {
        return; // frame no-JSON: ignorado sin romper el enlace
      }
      this.handleFrame(f);
    });

    ws.addEventListener("close", () => {
      this.stopHeartbeat();
      if (this.closedByUs) return;
      this.scheduleReconnect();
    });

    ws.addEventListener("error", () => {
      if (!this.closedByUs && this.statusValue !== "error") {
        this.setStatus("error", "error de socket (se reintenta)");
      }
    });
  }

  private handleFrame(f: Frame) {
    switch (f.op) {
      case "hello": {
        this.attempts = 0;
        this.setStatus("online", `enlace establecido (${String(f.msg ?? "puente OPC-UA")})`);
        return;
      }
      case "data": {
        this.statsValue.received++;
        const s = this.mapSample(f);
        if (!s) {
          this.statsValue.rejected++;
          return;
        }
        this.statsValue.accepted++;
        for (const cb of this.dataCbs) cb(s);
        if (this.statusValue !== "online") this.setStatus("online", "enlace establecido");
        return;
      }
      case "status": {
        if (f.state === "degraded") this.setStatus("degraded", String(f.msg ?? "fuente degradada"));
        return;
      }
      case "error": {
        this.setStatus("degraded", String(f.msg ?? "error reportado por el puente"));
        return;
      }
      default:
        return; // op desconocido: ignorar (compatibilidad hacia adelante)
    }
  }

  private mapSample(f: Frame): IngestSample | null {
    if (!this.tags || !f.values) return null;
    const rev: Record<string, VarKey> = {};
    for (const k of Object.keys(this.tags) as VarKey[]) rev[this.tags[k]] = k;
    const values: Partial<Record<VarKey, number>> = {};
    let any = false;
    for (const [nodeId, v] of Object.entries(f.values)) {
      const key = rev[nodeId];
      if (!key) continue; // nodo fuera del mapa: no es del pozo
      // estricto: solo números y cadenas numéricas. JSON.stringify serializa
      // NaN/Infinity como null — convertir null→0 inventaría datos falsos.
      let num = NaN;
      if (typeof v === "number") num = v;
      else if (typeof v === "string" && v.trim() !== "") num = Number(v);
      if (!Number.isFinite(num)) continue; // valor no numérico/NaN: se descarta
      values[key] = num;
      any = true;
    }
    return any ? {
      ts: typeof f.ts === "number" ? f.ts : Date.now(),
      timestampOrigin: f.timestampOrigin === "source" ? "source" : "receive",
      values,
    } : null;
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    this.hbTimer = setInterval(() => {
      const silent = Date.now() - this.lastMsgAt;
      if (silent > this.opts.heartbeatDeadMs) {
        // enlace zombi: cortar y reconectar
        this.setStatus("degraded", "sin latidos: reconectando");
        try {
          this.ws?.close();
        } catch {
          /* ya cerrado */
        }
        return;
      }
      if (silent > this.opts.heartbeatMs && this.statusValue === "online") {
        this.setStatus("degraded", "sin datos recientes (vigilancia de latidos)");
      }
    }, 5000);
  }

  private stopHeartbeat() {
    if (this.hbTimer) {
      clearInterval(this.hbTimer);
      this.hbTimer = null;
    }
  }

  private scheduleReconnect() {
    if (this.closedByUs) return;
    this.statsValue.reconnects++;
    this.attempts++;
    const delay = Math.min(
      this.opts.reconnectBaseMs * Math.pow(2, this.attempts - 1),
      this.opts.reconnectMaxMs,
    );
    this.setStatus("error", `enlace caído; reintento en ${Math.round(delay / 1000)} s`);
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => {
      if (!this.closedByUs) this.open();
    }, delay);
  }

  disconnect() {
    this.closedByUs = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopHeartbeat();
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        /* ya cerrado */
      }
      this.ws = null;
    }
    if (this.statusValue !== "idle") this.setStatus("idle", "fuente desconectada");
    this.attempts = 0;
  }
}
