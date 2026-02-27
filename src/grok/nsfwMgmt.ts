import type { GrokSettings } from "../settings";
import { getDynamicHeaders } from "./headers";

const NSFW_MGMT_API = "https://grok.com/auth_mgmt.AuthManagement/UpdateUserFeatureControls";
const ACCEPT_TOS_API = "https://accounts.x.ai/auth_mgmt.AuthManagement/SetTosAcceptedVersion";
const SET_BIRTH_API = "https://grok.com/rest/auth/set-birth-date";

export type GrpcStatus = { code: number; message: string };

export type NsfwEnableResult = {
  success: boolean;
  http_status: number;
  grpc_status?: number;
  grpc_message?: string | null;
  error?: string | null;
};

function encodeVarint(value: number): Uint8Array {
  const out: number[] = [];
  let v = Math.max(0, Math.floor(value));
  while (v >= 0x80) {
    out.push((v & 0x7f) | 0x80);
    v = Math.floor(v / 128);
  }
  out.push(v);
  return new Uint8Array(out);
}

function concatBytes(...parts: Array<Uint8Array | number[]>): Uint8Array {
  const arrays = parts.map((p) => (p instanceof Uint8Array ? p : new Uint8Array(p)));
  const total = arrays.reduce((sum, a) => sum + a.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const a of arrays) {
    out.set(a, offset);
    offset += a.byteLength;
  }
  return out;
}

function encodeGrpcWebFrame(data: Uint8Array): Uint8Array {
  const header = new Uint8Array(5);
  header[0] = 0x00;
  const len = data.byteLength >>> 0;
  header[1] = (len >>> 24) & 0xff;
  header[2] = (len >>> 16) & 0xff;
  header[3] = (len >>> 8) & 0xff;
  header[4] = len & 0xff;
  return concatBytes(header, data);
}

function toBodyInit(data: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(data.byteLength);
  copy.set(data);
  return copy.buffer;
}

function parseTrailerBlock(payload: Uint8Array): Record<string, string> {
  const text = new TextDecoder().decode(payload);
  const lines = text.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
  const out: Record<string, string> = {};
  for (const ln of lines) {
    const idx = ln.indexOf(":");
    if (idx === -1) continue;
    const key = ln.slice(0, idx).trim().toLowerCase();
    const value = ln.slice(idx + 1).trim();
    out[key] = key === "grpc-message" ? decodeURIComponent(value) : value;
  }
  return out;
}

function parseGrpcWebTrailers(body: Uint8Array, headers?: Headers): Record<string, string> {
  const trailers: Record<string, string> = {};
  let i = 0;
  while (i + 5 <= body.byteLength) {
    const flag = body[i]!;
    const len =
      ((body[i + 1]! << 24) | (body[i + 2]! << 16) | (body[i + 3]! << 8) | body[i + 4]!) >>> 0;
    i += 5;
    if (i + len > body.byteLength) break;
    const payload = body.slice(i, i + len);
    i += len;
    if (flag & 0x80) {
      Object.assign(trailers, parseTrailerBlock(payload));
    }
  }
  if (headers) {
    const hStatus = headers.get("grpc-status");
    const hMsg = headers.get("grpc-message");
    if (hStatus && !trailers["grpc-status"]) trailers["grpc-status"] = hStatus.trim();
    if (hMsg && !trailers["grpc-message"]) trailers["grpc-message"] = decodeURIComponent(hMsg.trim());
  }
  return trailers;
}

function grpcStatusFromTrailers(trailers: Record<string, string>): GrpcStatus {
  const raw = String(trailers["grpc-status"] ?? "").trim();
  const msg = String(trailers["grpc-message"] ?? "").trim();
  const code = Number.parseInt(raw, 10);
  return { code: Number.isFinite(code) ? code : -1, message: msg };
}

function buildNsfwProtobuf(featureKey: string): Uint8Array {
  const nameBytes = new TextEncoder().encode(featureKey);

  // Mirrors Python implementation:
  // inner = 0x0a + len(name) + name
  // protobuf = 0x0a 0x02 0x10 0x01 0x12 + len(inner) + inner
  const inner = concatBytes([0x0a], encodeVarint(nameBytes.byteLength), nameBytes);
  const protobuf = concatBytes([0x0a, 0x02, 0x10, 0x01, 0x12], encodeVarint(inner.byteLength), inner);
  return protobuf;
}

export async function enableNsfwFeature(args: {
  cookie: string;
  settings: GrokSettings;
  featureKey: string;
  timeoutMs?: number;
}): Promise<GrpcStatus> {
  const controller = new AbortController();
  const timeoutMs = Math.max(1, Math.floor(args.timeoutMs ?? 60000));
  const timeoutId = setTimeout(() => controller.abort("timeout"), timeoutMs);

  try {
    const headers = getDynamicHeaders(args.settings, "/auth_mgmt.AuthManagement/UpdateUserFeatureControls");
    headers.Cookie = args.cookie;
    headers.Referer = "https://grok.com/?_s=data";
    headers.Accept = "*/*";
    headers["Content-Type"] = "application/grpc-web+proto";
    headers["Sec-Fetch-Dest"] = "empty";
    headers["x-grpc-web"] = "1";
    headers["x-user-agent"] = "connect-es/2.1.1";
    headers["Cache-Control"] = "no-cache";
    headers.Pragma = "no-cache";

    const protobuf = buildNsfwProtobuf(args.featureKey);
    const body = encodeGrpcWebFrame(protobuf);

    const resp = await fetch(NSFW_MGMT_API, {
      method: "POST",
      headers,
      body: toBodyInit(body),
      signal: controller.signal,
    });

    const buf = new Uint8Array(await resp.arrayBuffer());
    if (!resp.ok) {
      const txt = new TextDecoder().decode(buf.slice(0, Math.min(buf.byteLength, 512)));
      throw new Error(`Upstream ${resp.status}: ${txt}`);
    }

    const trailers = parseGrpcWebTrailers(buf, resp.headers);
    return grpcStatusFromTrailers(trailers);
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function acceptTos(args: {
  cookie: string;
  settings: GrokSettings;
  timeoutMs?: number;
}): Promise<GrpcStatus> {
  const controller = new AbortController();
  const timeoutMs = Math.max(1, Math.floor(args.timeoutMs ?? 60000));
  const timeoutId = setTimeout(() => controller.abort("timeout"), timeoutMs);

  try {
    const headers = getDynamicHeaders(args.settings, "/auth_mgmt.AuthManagement/SetTosAcceptedVersion");
    headers.Cookie = args.cookie;
    headers.Origin = "https://accounts.x.ai";
    headers.Referer = "https://accounts.x.ai/accept-tos";
    headers.Accept = "*/*";
    headers["Content-Type"] = "application/grpc-web+proto";
    headers["Sec-Fetch-Dest"] = "empty";
    headers["x-grpc-web"] = "1";
    headers["x-user-agent"] = "connect-es/2.1.1";
    headers["Cache-Control"] = "no-cache";
    headers.Pragma = "no-cache";

    const body = encodeGrpcWebFrame(new Uint8Array([0x10, 0x01]));
    const resp = await fetch(ACCEPT_TOS_API, {
      method: "POST",
      headers,
      body: toBodyInit(body),
      signal: controller.signal,
    });

    const buf = new Uint8Array(await resp.arrayBuffer());
    if (!resp.ok) {
      const txt = new TextDecoder().decode(buf.slice(0, Math.min(buf.byteLength, 512)));
      throw new Error(`Upstream ${resp.status}: ${txt}`);
    }
    const trailers = parseGrpcWebTrailers(buf, resp.headers);
    return grpcStatusFromTrailers(trailers);
  } finally {
    clearTimeout(timeoutId);
  }
}

function randomBirthDateIso(): string {
  const today = new Date();
  const year = today.getUTCFullYear() - Math.floor(20 + Math.random() * 29); // 20..48
  const month = Math.floor(1 + Math.random() * 12);
  const day = Math.floor(1 + Math.random() * 28);
  const hour = Math.floor(Math.random() * 24);
  const minute = Math.floor(Math.random() * 60);
  const second = Math.floor(Math.random() * 60);
  const ms = Math.floor(Math.random() * 1000);
  const pad = (n: number, len = 2) => String(n).padStart(len, "0");
  return `${pad(year, 4)}-${pad(month)}-${pad(day)}T${pad(hour)}:${pad(minute)}:${pad(second)}.${pad(ms, 3)}Z`;
}

export async function setBirthDate(args: {
  cookie: string;
  settings: GrokSettings;
  timeoutMs?: number;
}): Promise<{ ok: true; status: number }> {
  const controller = new AbortController();
  const timeoutMs = Math.max(1, Math.floor(args.timeoutMs ?? 60000));
  const timeoutId = setTimeout(() => controller.abort("timeout"), timeoutMs);

  try {
    const headers = getDynamicHeaders(args.settings, "/rest/auth/set-birth-date");
    headers.Cookie = args.cookie;
    headers.Referer = "https://grok.com/?_s=home";
    headers["Content-Type"] = "application/json";

    const payload = JSON.stringify({ birthDate: randomBirthDateIso() });
    const resp = await fetch(SET_BIRTH_API, {
      method: "POST",
      headers,
      body: payload,
      signal: controller.signal,
    });
    if (resp.status === 200 || resp.status === 204) return { ok: true, status: resp.status };
    const txt = await resp.text().catch(() => "");
    throw new Error(`Upstream ${resp.status}: ${txt.slice(0, 200)}`);
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function enableNsfw(args: {
  cookie: string;
  settings: GrokSettings;
  featureKey: string;
  timeoutMs?: number;
}): Promise<NsfwEnableResult> {
  try {
    const timeoutMs = args.timeoutMs ?? 60000;
    const tos = await acceptTos({ cookie: args.cookie, settings: args.settings, timeoutMs });
    if (![0, -1].includes(tos.code)) {
      return { success: false, http_status: 502, grpc_status: tos.code, grpc_message: tos.message, error: "Accept ToS failed" };
    }

    await setBirthDate({ cookie: args.cookie, settings: args.settings, timeoutMs });

    const st = await enableNsfwFeature({
      cookie: args.cookie,
      settings: args.settings,
      featureKey: args.featureKey,
      timeoutMs,
    });
    const success = st.code === 0 || st.code === -1;
    return {
      success,
      http_status: 200,
      grpc_status: st.code,
      grpc_message: st.message || null,
      error: success ? null : "NSFW enable failed",
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { success: false, http_status: 0, error: msg.slice(0, 200) };
  }
}
