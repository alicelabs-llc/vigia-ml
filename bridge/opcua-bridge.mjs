#!/usr/bin/env node
// ---------------------------------------------------------------------------
// VIGÍA ML · Puente OPC-UA → WebSocket (v0.12.0)
// El navegador no puede hablar el protocolo binario OPC-UA: este puente corre
// en un nodo de red con acceso al servidor de automatización (SCADA/RTU/PLC)
// y reenvía las lecturas a la consola por WebSocket en frames JSON.
//
// Uso:
//   OPC_ENDPOINT="opc.tcp://servidor:4840" node opcua-bridge.mjs --map mapa.json
//   VIGIA_DEMO=1 node opcua-bridge.mjs            # modo demo sin servidor real
// Por seguridad escucha solo en 127.0.0.1. No expongas el puerto a una red
// industrial o pública sin un gateway autenticado y cifrado.
//
// Frames WebSocket (JSON una línea por mensaje):
//   puente→cliente: {"op":"hello","msg":"..."}
//                   {"op":"data","ts":169...,"values":{"PT-101":123.4,...}}
//                   {"op":"status","state":"degraded","msg":"..."}
//                   {"op":"error","msg":"..."}
//   cliente→puente: {"op":"subscribe","tags":{"pt":"PT-101",...},"intervalMs":15000}
//
// El mapa de nodos (mapa.json) vincula cada variable de VIGÍA con su NodeId:
//   { "pt": "ns=2;s=Canal1.Dispositivo1.Tag1", "pc": "...", ... }
// En OPC-UA real el mapa del servidor (--map / TAGS) es obligatorio y autoritativo.

// ---------------------------------------------------------------------------

import { WebSocketServer } from "ws";
import { CLIENT_TAGS, normalizeDataValue, validateClientTagMap, validateTagMap } from "./config.mjs";

// ----------------------------- configuración --------------------------------
const args = process.argv.slice(2);
const argOf = (name, def = null) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : def;
};

const PORT = Number(argOf("--port", process.env.WS_PORT ?? 8082));
const HOST = argOf("--host", process.env.WS_HOST ?? "127.0.0.1");
if (!["127.0.0.1", "::1", "localhost"].includes(HOST)) {
  console.error("[bridge] host remoto rechazado: usa un reverse proxy/VPN que conecte al loopback local");
  process.exit(1);
}
const ENDPOINT = argOf("--endpoint", process.env.OPC_ENDPOINT ?? null);
const DEMO = process.env.VIGIA_DEMO === "1" || !ENDPOINT;
const INTERVAL = Math.max(5000, Number(argOf("--interval", 15000)));

let fileMap = {};
const mapPath = argOf("--map", process.env.TAGMAP_FILE ?? null);
try {
  if (mapPath) {
    fileMap = JSON.parse(await import("node:fs").then((fs) => fs.readFileSync(mapPath, "utf8")));
  } else if (process.env.TAGS) {
    fileMap = JSON.parse(process.env.TAGS);
  }
  if (Object.keys(fileMap).length) fileMap = validateTagMap(fileMap, { requireAll: !DEMO });
  if (!DEMO) fileMap = validateTagMap(fileMap, { requireAll: true });
} catch (e) {
  console.error(`[bridge] mapa OPC-UA inválido: ${e.message}`);
  process.exit(1);
}

// valores sintéticos del modo demo (rango físico de un pozo de gas)
const DEMO_VALUES = {
  pt: () => 980 + Math.sin(Date.now() / 60000) * 40 + Math.random() * 8,
  pc: () => 1250 + Math.sin(Date.now() / 90000) * 30 + Math.random() * 6,
  pl: () => 640 + Math.random() * 10,
  temp: () => 42 + Math.random() * 2.5,
  q: () => 8200 + Math.sin(Date.now() / 45000) * 350 + Math.random() * 60,
  choke: () => 68 + Math.random() * 2,
};

console.log(`[bridge] VIGÍA OPC-UA · modo ${DEMO ? "DEMO (VIGIA_DEMO=1)" : "OPC-UA real"}`);
if (!DEMO) console.log(`[bridge] endpoint: ${ENDPOINT.replace(/\/\/[^@]*@/, "//***@")}`);

// --------------------------- sesión OPC-UA ----------------------------------
// node-opcua se importa en diferido: en modo demo no hace falta instalarlo.
async function startOpcua(clientMap) {
  const opcua = await import("node-opcua");
  const client = opcua.OPCUAClient.create({ applicationName: "vigia-ml-bridge" });
  await client.connect(ENDPOINT);
  console.log("[bridge] conectado al servidor OPC-UA");
  const session = await client.createSession();
  const sub = await session.createSubscription2({
    requestedPublishingInterval: INTERVAL,
    requestedLifetimeCount: 100,
    requestedMaxKeepAliveCount: 10,
    maxNotificationsPerPublish: 64,
    publishingEnabled: true,
    priority: 10,
  });

  const itemsToMonitor = Object.entries(clientMap).map(([key, nodeId]) => ({
    key,
    nodeId,
    attributeId: opcua.AttributeIds.Value,
  }));

  const Broadcast = (fn) => {
    for (const c of wss.clients) if (c.readyState === 1) fn(c);
  };

  await Promise.all(
    itemsToMonitor.map((it) =>
      sub.monitor(
        { nodeId: it.nodeId, attributeId: it.attributeId },
        { samplingInterval: INTERVAL, queueSize: 4 },
        opcua.TimestampsToReturn.Both,
        (err, _mon) => {
          if (err) console.error(`[bridge] monitor ${it.key}: ${err.message}`);
        },
      ).on("changed", (dv) => {
        const sample = normalizeDataValue(dv);
        if (!sample) {
          Broadcast((c) => c.send(JSON.stringify({
            op: "status",
            state: "degraded",
            msg: `calidad o valor inválido para ${it.key}; lectura descartada`,
          })));
          return;
        }
        Broadcast((c) => c.send(JSON.stringify({
          op: "data",
          ts: sample.ts,
          timestampOrigin: sample.timestampOrigin,
          values: { [CLIENT_TAGS[it.key]]: sample.value },
        })));
      }),
    ),
  );

  const reconnect = () => {
    Broadcast((c) => c.send(JSON.stringify({ op: "status", state: "degraded", msg: "sesión OPC-UA caída; reconectando" })));
    setTimeout(() => startOpcua(clientMap).catch((e) => {
      console.error(`[bridge] reintento falló: ${e.message}`);
      setTimeout(reconnect, 15000);
    }), 5000);
  };
  client.on("connection_lost", reconnect);
  client.on("backoff", () => {});
  return { session, client, reconnect };
}

// --------------------------- servidor WebSocket -----------------------------
import { createServer } from "node:http";
const httpServer = createServer((_req, res) => {
  res.writeHead(426);
  res.end("VIGÍA OPC-UA bridge: use WebSocket\n");
});
const wss = new WebSocketServer({ server: httpServer });

wss.on("connection", (socket) => {
  console.log(`[bridge] cliente conectado (${wss.clients.size} activos)`);
  socket.send(JSON.stringify({ op: "hello", msg: DEMO ? "puente demo" : "puente OPC-UA" }));
  let timer = null;
  let clientMap = { ...fileMap };

  socket.on("message", (raw) => {
    let f;
    try {
      f = JSON.parse(raw.toString());
    } catch {
      return;
    }
    if (f.op === "subscribe" && f.tags && typeof f.tags === "object") {
      try {
        // El navegador usa alias estables (PT-101, FT-301); el servidor
        // traduce esos alias a NodeIds físicos mediante --map.
        const requestedMap = validateClientTagMap(f.tags);
        clientMap = requestedMap;
      } catch (e) {
        socket.send(JSON.stringify({ op: "error", msg: `suscripción rechazada: ${e.message}` }));
        return;
      }
      if (timer) clearInterval(timer);
      timer = setInterval(() => {
        if (!DEMO) return; // en modo real empujan los 'changed' de la suscripción
        try {
          // un frame combinado por ciclo, etiquetado con el NodeId que pidió el cliente
          const values = {};
          for (const [key, tag] of Object.entries(clientMap)) {
            if (typeof DEMO_VALUES[key] === "function") values[tag] = Number(DEMO_VALUES[key]().toFixed(2));
          }
          socket.send(JSON.stringify({ op: "data", ts: Date.now(), values }));
        } catch (e) {
          socket.send(JSON.stringify({ op: "error", msg: `lectura falló: ${e.message}` }));
        }
      }, INTERVAL);
      socket.send(
        JSON.stringify({
          op: "status",
          state: "ok",
          msg: `suscrito a ${Object.keys(clientMap).length} variables @ ${INTERVAL / 1000}s`,
        }),
      );
    }
  });

  socket.on("close", () => {
    if (timer) clearInterval(timer);
    console.log(`[bridge] cliente desconectado (${wss.clients.size} activos)`);
  });
});

httpServer.listen(PORT, HOST, () => {
  console.log(`[bridge] WebSocket en ws://${HOST}:${PORT} (intervalo ${INTERVAL / 1000}s)`);
  if (!DEMO) {
    startOpcua(fileMap).catch((e) => {
      console.error(`[bridge] no se pudo conectar al servidor OPC-UA: ${e.message}`);
      process.exit(2);
    });
  }
});

const shutdown = () => {
  console.log("\n[bridge] cierre ordenado");
  for (const c of wss.clients) c.close();
  httpServer.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 2000);
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
