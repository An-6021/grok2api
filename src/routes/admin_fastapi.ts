import { Hono } from "hono";
import type { Env } from "../env";
import { dbAll } from "../db";
import { nowMs } from "../utils/time";
import { getFastApiConfig, setFastApiConfig } from "../fastapiConfig";
import { requireAppKeyAuth } from "../auth";

type TokenRow = {
  token: string;
  token_type: "sso" | "ssoSuper";
  status: string;
  remaining_queries: number;
  tags: string;
  note: string;
  created_time: number;
};

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function parseTags(raw: unknown): string[] {
  if (Array.isArray(raw)) {
    return raw.map((x) => String(x ?? "").trim()).filter(Boolean);
  }
  if (typeof raw === "string" && raw.trim()) {
    try {
      const parsed = JSON.parse(raw) as unknown;
      if (Array.isArray(parsed)) {
        return parsed.map((x) => String(x ?? "").trim()).filter(Boolean);
      }
    } catch {
      // ignore
    }
  }
  return [];
}

function normalizeStatus(raw: unknown): "active" | "cooling" | "expired" | "disabled" {
  const s = String(raw ?? "active").trim().toLowerCase();
  if (s === "active" || s === "cooling" || s === "expired" || s === "disabled") return s;
  return "active";
}

function normalizeToken(raw: unknown): string {
  const t = String(raw ?? "").trim();
  if (!t) return "";
  return t.startsWith("sso=") ? t.slice(4).trim() : t;
}

function defaultQuotaForPool(pool: "ssoBasic" | "ssoSuper"): number {
  return pool === "ssoSuper" ? 140 : 80;
}

export const adminFastApiRoutes = new Hono<{ Bindings: Env }>();

adminFastApiRoutes.get("/verify", requireAppKeyAuth, (c) => {
  return c.json({ status: "success" });
});

adminFastApiRoutes.get("/storage", requireAppKeyAuth, (c) => {
  return c.json({ type: "d1" });
});

adminFastApiRoutes.get("/config", requireAppKeyAuth, async (c) => {
  const cfg = await getFastApiConfig(c.env);
  return c.json(cfg);
});

adminFastApiRoutes.post("/config", requireAppKeyAuth, async (c) => {
  try {
    const body = (await c.req.json()) as unknown;
    await setFastApiConfig(c.env, body);
    return c.json({ status: "success", message: "配置已更新" });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

adminFastApiRoutes.get("/tokens", requireAppKeyAuth, async (c) => {
  const rows = await dbAll<TokenRow>(
    c.env.grok2api,
    "SELECT token, token_type, status, remaining_queries, tags, note, created_time FROM tokens ORDER BY created_time DESC",
  );
  const out: Record<string, unknown[]> = { ssoBasic: [], ssoSuper: [] };
  for (const r of rows) {
    const pool = r.token_type === "ssoSuper" ? "ssoSuper" : "ssoBasic";
    const quota = Number.isFinite(r.remaining_queries) && r.remaining_queries > 0 ? r.remaining_queries : 0;
    out[pool]!.push({
      token: r.token,
      status: normalizeStatus(r.status),
      quota,
      note: r.note ?? "",
      use_count: 0,
      tags: parseTags(r.tags),
    });
  }
  return c.json(out);
});

adminFastApiRoutes.post("/tokens", requireAppKeyAuth, async (c) => {
  try {
    const body = (await c.req.json()) as unknown;
    if (!isPlainObject(body)) return c.json({ detail: "Invalid payload" }, 400);

    const pools: Array<{ pool: "ssoBasic" | "ssoSuper"; token_type: "sso" | "ssoSuper" }> = [
      { pool: "ssoBasic", token_type: "sso" },
      { pool: "ssoSuper", token_type: "ssoSuper" },
    ];

    const rows: Array<{
      token: string;
      token_type: "sso" | "ssoSuper";
      created_time: number;
      remaining_queries: number;
      heavy_remaining_queries: number;
      status: string;
      tags_json: string;
      note: string;
    }> = [];
    const now = nowMs();
    let index = 0;

    for (const p of pools) {
      const listRaw = body[p.pool];
      if (!Array.isArray(listRaw)) continue;
      for (const item of listRaw) {
        const token = normalizeToken(isPlainObject(item) ? item.token : item);
        if (!token) continue;
        const quotaRaw = isPlainObject(item) ? item.quota : undefined;
        const quotaNumber = Number(quotaRaw);
        const quota = Number.isFinite(quotaNumber) ? Math.max(0, Math.floor(quotaNumber)) : defaultQuotaForPool(p.pool);
        const status = normalizeStatus(isPlainObject(item) ? item.status : "active");
        const tags = parseTags(isPlainObject(item) ? item.tags : []);
        const note = isPlainObject(item) ? String(item.note ?? "").trim() : "";

        const created = now - index;
        index += 1;

        rows.push({
          token,
          token_type: p.token_type,
          created_time: created,
          remaining_queries: quota,
          heavy_remaining_queries: -1,
          status,
          tags_json: JSON.stringify(tags),
          note,
        });
      }
    }

    // D1 (and the DO-backed SQLite) does not allow explicit BEGIN/COMMIT.
    // `db.batch([...])` already executes in a single atomic transaction.
    const stmts: D1PreparedStatement[] = [c.env.grok2api.prepare("DELETE FROM tokens")];

    const chunkSize = 100; // keep bind params under SQLite limits and stmt count under D1 batch limits
    for (let i = 0; i < rows.length; i += chunkSize) {
      const chunk = rows.slice(i, i + chunkSize);
      const valuesSql = chunk.map(() => "(?,?,?,?,?,?,0,NULL,NULL,NULL,?,?)").join(",");
      const sql =
        "INSERT OR REPLACE INTO tokens(token, token_type, created_time, remaining_queries, heavy_remaining_queries, status, failed_count, cooldown_until, last_failure_time, last_failure_reason, tags, note) VALUES " +
        valuesSql;
      const params: unknown[] = [];
      for (const r of chunk) {
        params.push(
          r.token,
          r.token_type,
          r.created_time,
          r.remaining_queries,
          r.heavy_remaining_queries,
          r.status,
          r.tags_json,
          r.note,
        );
      }
      stmts.push(c.env.grok2api.prepare(sql).bind(...params));
    }

    await c.env.grok2api.batch(stmts);

    return c.json({ status: "success", message: "Token 已更新" });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

// ======================================================================
// Not implemented in phase 1. Keep endpoints predictable for the UI.
// ======================================================================

function notImplemented(): Response {
  return new Response(JSON.stringify({ status: "error", error: "Not implemented in Cloudflare Workers build" }), {
    status: 501,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

adminFastApiRoutes.all("/tokens/refresh", requireAppKeyAuth, () => notImplemented());
adminFastApiRoutes.all("/tokens/refresh/async", requireAppKeyAuth, () => notImplemented());
adminFastApiRoutes.all("/tokens/nsfw/enable", requireAppKeyAuth, () => notImplemented());
adminFastApiRoutes.all("/tokens/nsfw/enable/async", requireAppKeyAuth, () => notImplemented());

adminFastApiRoutes.all("/cache", requireAppKeyAuth, () => notImplemented());
adminFastApiRoutes.all("/cache/list", requireAppKeyAuth, () => notImplemented());
adminFastApiRoutes.all("/cache/clear", requireAppKeyAuth, () => notImplemented());
adminFastApiRoutes.all("/cache/item/delete", requireAppKeyAuth, () => notImplemented());
adminFastApiRoutes.all("/cache/online/load/async", requireAppKeyAuth, () => notImplemented());
adminFastApiRoutes.all("/cache/online/clear/async", requireAppKeyAuth, () => notImplemented());
adminFastApiRoutes.all("/cache/online/clear", requireAppKeyAuth, () => notImplemented());
