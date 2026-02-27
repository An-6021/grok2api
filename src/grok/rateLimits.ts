import type { GrokSettings } from "../settings";
import { getDynamicHeaders } from "./headers";
import { toRateLimitModel } from "./models";

const RATE_LIMIT_API = "https://grok.com/rest/rate-limits";

export async function fetchRateLimits(
  cookie: string,
  settings: GrokSettings,
  model: string,
): Promise<{ ok: true; status: number; data: Record<string, unknown> } | { ok: false; status: number; body: string }> {
  const rateModel = toRateLimitModel(model);
  const headers = getDynamicHeaders(settings, "/rest/rate-limits");
  headers.Cookie = cookie;
  const body = JSON.stringify({ requestKind: "DEFAULT", modelName: rateModel });

  const resp = await fetch(RATE_LIMIT_API, { method: "POST", headers, body });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    return { ok: false, status: resp.status, body: text.slice(0, 400) };
  }
  const data = (await resp.json()) as Record<string, unknown>;
  return { ok: true, status: resp.status, data };
}

export async function checkRateLimits(
  cookie: string,
  settings: GrokSettings,
  model: string,
): Promise<Record<string, unknown> | null> {
  const res = await fetchRateLimits(cookie, settings, model);
  return res.ok ? res.data : null;
}
