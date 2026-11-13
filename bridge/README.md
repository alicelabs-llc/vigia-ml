# Puente OPC-UA → WebSocket de VIGÍA ML

El navegador **no puede hablar el protocolo binario OPC-UA** (TCP 4840, codificación DER): ningún despliegue serio expone un SCADA directo al browser. La arquitectura de integración de VIGÍA ML es un **puente** de un solo sentido que corre en un equipo con acceso al servidor de automatización y reenvía las lecturas a la consola por WebSocket en frames JSON.

```
┌──────────┐  OPC-UA    ┌─────────────────┐  WebSocket  ┌────────────────────┐
│ SCADA /  │ ─────────▶ │ opcua-bridge.mjs│ ──────────▶ │ vigia-ml (browser) │
│ RTU/PLC  │  opc.tcp   │  (nodo de red)  │   :8082     │  PWA · offline     │
└──────────┘  :4840     └─────────────────┘             └────────────────────┘
```

## Protocolo (frames JSON, una línea por mensaje)

| Dirección | Frame | Significado |
|---|---|---|
| puente → consola | `{"op":"hello","msg":"..."}` | saludo tras conectar |
| puente → consola | `{"op":"data","ts":...,"values":{"PT-101":123.4}}` | lecturas etiquetadas por alias lógico; `ts` usa timestamp de origen cuando existe |
| puente → consola | `{"op":"status","state":"degraded","msg":"..."}` | fuente degradada |
| puente → consola | `{"op":"error","msg":"..."}` | error de lectura |
| consola → puente | `{"op":"subscribe","tags":{"pt":"PT-101","pc":"PT-102",...},"intervalMs":15000}` | mapa de variables |

El puente descarta lecturas OPC-UA con calidad no válida y valores no numéricos/no finitos antes de emitirlas. El cliente vuelve a sanear cada muestra antes de incorporarla al buffer.

## Modo demo (sin ningún servidor OPC-UA)

```bash
cd bridge
npm install
npm run start:demo          # VIGIA_DEMO=1 · solo necesita `ws`
```

Con la consola abierta (`https://…/vigia-ml/`), en **Fuente de datos** elige
"Puente OPC-UA", introduce `ws://localhost:8082` y pulsa **Conectar**. Verás
las telemetrías sintéticas del puente alimentar el pozo seleccionado en tiempo
real. (Nota: desde la demo publicada en GitHub Pages el navegador bloquea
`ws://` inseguro en página `https://`; usa la consola servida por `npm run dev`
o un puente detrás de `wss://` — ver Seguridad.)

## Modo real (servidor OPC-UA)

```bash
cd bridge
npm install
OPC_ENDPOINT="opc.tcp://servidor:4840" node opcua-bridge.mjs \
  --map mapa.json \
  --port 8082 \
  --interval 15000
```

`mapa.json` vincula cada variable de VIGÍA con su NodeId en tu servidor:

```json
{
  "pt": "ns=2;s=Canal1.Dispositivo1.PT101",
  "pc": "ns=2;s=Canal1.Dispositivo1.PT102",
  "pl": "ns=2;s=Canal1.Dispositivo1.PT108",
  "temp": "ns=2;s=Canal1.Dispositivo1.TT201",
  "q": "ns=2;s=Canal1.Dispositivo1.FT301",
  "choke": "ns=2;s=Canal1.Dispositivo1.FV401"
}
```

Variables: `pt` presión de tubing · `pc` presión de casing · `pl` presión de
línea · `temp` temperatura de cabezal · `q` caudal de gas · `choke` apertura
de choke. El cliente descarta nodos fuera del mapa.

### Variables de entorno equivalentes

`OPC_ENDPOINT`, `TAGMAP_FILE` (o `TAGS` como JSON), `WS_PORT`, `VIGIA_DEMO=1`. El host permanece limitado a loopback; usa un proxy seguro para acceso remoto.

## Seguridad (léelo antes de producción)

- El puente escucha por defecto solo en `127.0.0.1`; esto evita publicar un socket SCADA en todas las interfaces por accidente.
- El puente rechaza cualquier host que no sea loopback. Para acceso desde otra máquina, usa un reverse proxy local con `wss://` (TLS), autenticación y reglas de firewall, o una VPN que termine en ese proxy. No expongas el proceso Node directamente.
- Proporciona `OPC_ENDPOINT` mediante el entorno del proceso; evita incluir usuario/contraseña en argumentos de línea de comandos que puedan aparecer en listados de procesos.
- Usa credenciales OPC-UA dedicadas de solo lectura para VIGÍA y un mapa de nodos explícito. En modo real, el mapa del servidor es obligatorio y autoritativo; el mapa del navegador debe usar los seis alias lógicos documentados.
- El mapa de nodos del servidor debe contener las seis variables `pt`, `pc`, `pl`, `temp`, `q` y `choke`, con NodeIds OPC-UA válidos. Se valida al arrancar. El navegador usa alias lógicos estables (PT-101, PT-102, PT-108, TT-201, FT-301 y FV-401); el puente traduce cada alias al NodeId configurado y no revela NodeIds físicos al navegador.
- Cada muestra pasa por `sanitizeSample()` en el navegador: valores fuera de
  rango físico se sustituyen antes de llegar a los modelos.

## Licencia

AL-1.0 (AliceLabs Source-Available) — mismo término que el proyecto principal;
ver `../LICENSE-AL-1.0`.
