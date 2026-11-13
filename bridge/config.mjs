// Validación estricta del mapa OPC-UA de VIGÍA.
// Mantener este módulo sin dependencias permite probar la configuración sin
// conectarse a un PLC/SCADA real.
export const REQUIRED_TAGS = Object.freeze(["pt", "pc", "pl", "temp", "q", "choke"]);

const NODE_ID = /^(?:ns=\d+;)?(?:i=\d+|s=.+|g=\{?[0-9a-f-]{36}\}?|b=[A-Za-z0-9+/]+={0,2})$/i;

export function validateTagMap(input, { requireAll = false } = {}) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new TypeError("el mapa debe ser un objeto JSON");
  }

  const keys = Object.keys(input);
  const unknown = keys.filter((key) => !REQUIRED_TAGS.includes(key));
  if (unknown.length) throw new Error(`variables desconocidas: ${unknown.join(", ")}`);

  const result = {};
  for (const key of keys) {
    const nodeId = input[key];
    if (typeof nodeId !== "string" || !NODE_ID.test(nodeId.trim())) {
      throw new Error(`NodeId inválido para ${key}`);
    }
    result[key] = nodeId.trim();
  }

  if (requireAll) {
    const missing = REQUIRED_TAGS.filter((key) => !result[key]);
    if (missing.length) throw new Error(`faltan variables obligatorias: ${missing.join(", ")}`);
  }
  if (!Object.keys(result).length) throw new Error("el mapa no contiene variables");
  return result;
}

/** Normalize an OPC-UA DataValue; null means discard it. */
export function normalizeDataValue(dataValue, now = Date.now()) {
  if (dataValue?.statusCode && typeof dataValue.statusCode.isGood === "function" && !dataValue.statusCode.isGood()) return null;
  const value = dataValue?.value?.value;
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  const sourceTs = dataValue.sourceTimestamp instanceof Date ? dataValue.sourceTimestamp.getTime() : NaN;
  const hasSourceTs = Number.isFinite(sourceTs);
  return {
    value,
    ts: hasSourceTs ? sourceTs : now,
    timestampOrigin: hasSourceTs ? "source" : "receive",
  };
}

/** Alias de telemetría estable que espera el cliente de VIGÍA. */
export const CLIENT_TAGS = Object.freeze({
  pt: "PT-101",
  pc: "PT-102",
  pl: "PT-108",
  temp: "TT-201",
  q: "FT-301",
  choke: "FV-401",
});

/** Valida el mapa lógico del navegador (alias), distinto al mapa físico NodeId. */
export function validateClientTagMap(input) {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new TypeError("el mapa del cliente debe ser un objeto JSON");
  }
  const keys = Object.keys(input);
  const unknown = keys.filter((key) => !REQUIRED_TAGS.includes(key));
  if (unknown.length) throw new Error(`variables desconocidas: ${unknown.join(", ")}`);
  const missing = REQUIRED_TAGS.filter((key) => input[key] !== CLIENT_TAGS[key]);
  if (missing.length) throw new Error(`alias de cliente ausentes o incorrectos: ${missing.join(", ")}`);
  return { ...CLIENT_TAGS };
}
