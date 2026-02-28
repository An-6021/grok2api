import { Hono } from "hono";
import type { Env } from "../env";
import { dbAll, dbFirst, dbRun } from "../db";
import { nowMs } from "../utils/time";
import { getFastApiConfig, setFastApiConfig } from "../fastapiConfig";
import { requireAppKeyAuth } from "../auth";
import { getSettings, normalizeCfCookie } from "../settings";
import { fetchRateLimits } from "../grok/rateLimits";
import { enableNsfw } from "../grok/nsfwMgmt";
import { applyCooldown, recordTokenFailure } from "../repo/tokens";
import {
  createBatchTask,
  deleteBatchTask,
  getBatchTask,
  isBatchCancelled,
  markBatchCancelled,
  updateBatchTask,
  type BatchTask,
} from "../repo/batchTasks";
import {
  deleteCacheRow,
  deleteCacheRows,
  getCacheSizeBytes,
  listCacheRowsByType,
  listOldestRows,
} from "../repo/cache";

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

function buildCookie(token: string, cf: string): string {
  return cf ? `sso-rw=${token};sso=${token};${cf}` : `sso-rw=${token};sso=${token}`;
}

function maskToken(token: string): string {
  const t = String(token || "").trim();
  if (!t) return "";
  if (t.length > 20) return `${t.slice(0, 8)}...${t.slice(-8)}`;
  return t;
}

function parseTokenList(payload: unknown): string[] {
  if (!isPlainObject(payload)) return [];
  const out: string[] = [];
  const tokenRaw = payload.token;
  if (typeof tokenRaw === "string" && tokenRaw.trim()) out.push(tokenRaw.trim());
  const tokensRaw = payload.tokens;
  if (Array.isArray(tokensRaw)) {
    for (const item of tokensRaw) {
      const t = String(item ?? "").trim();
      if (t) out.push(t);
    }
  }
  return [...new Set(out.map(normalizeToken).filter(Boolean))];
}

function remainingFromRateLimits(data: Record<string, unknown>): number | null {
  const a = Number((data as any)?.remainingTokens);
  if (Number.isFinite(a)) return Math.max(0, Math.floor(a));
  const b = Number((data as any)?.remainingQueries);
  if (Number.isFinite(b)) return Math.max(0, Math.floor(b));
  return null;
}

async function setTokenQuota(args: {
  env: Env;
  token: string;
  remaining: number;
  heavyRemaining?: number;
}) {
  const status = args.remaining === 0 ? "cooling" : "active";
  const parts: string[] = [
    "remaining_queries = ?",
    "status = ?",
    "failed_count = 0",
    "cooldown_until = NULL",
    "last_failure_time = NULL",
    "last_failure_reason = NULL",
  ];
  const params: unknown[] = [args.remaining, status];
  if (typeof args.heavyRemaining === "number") {
    parts.unshift("heavy_remaining_queries = ?");
    params.unshift(args.heavyRemaining);
  }
  params.push(args.token);
  await dbRun(args.env.grok2api, `UPDATE tokens SET ${parts.join(", ")} WHERE token = ?`, params);
}

async function refreshOneToken(args: {
  env: Env;
  token: string;
  settings: Awaited<ReturnType<typeof getSettings>>["grok"];
  cf: string;
  tokenType: "sso" | "ssoSuper";
}): Promise<boolean> {
  const cookie = buildCookie(args.token, args.cf);

  const base = await fetchRateLimits(cookie, args.settings, "grok-4");
  if (!base.ok) {
    await recordTokenFailure(args.env.grok2api, args.token, base.status, base.body);
    await applyCooldown(args.env.grok2api, args.token, base.status);
    return false;
  }

  const remaining = remainingFromRateLimits(base.data);
  if (remaining === null) {
    await recordTokenFailure(args.env.grok2api, args.token, 500, "missing_remaining");
    await applyCooldown(args.env.grok2api, args.token, 500);
    return false;
  }

  let heavyRemaining: number | undefined;
  if (args.tokenType === "ssoSuper") {
    const heavy = await fetchRateLimits(cookie, args.settings, "grok-4-heavy");
    if (heavy.ok) {
      const parsed = remainingFromRateLimits(heavy.data);
      if (parsed !== null) heavyRemaining = parsed;
    }
  }

  await setTokenQuota({
    env: args.env,
    token: args.token,
    remaining,
    ...(heavyRemaining !== undefined ? { heavyRemaining } : {}),
  });
  return true;
}

function sseHeaders(): Record<string, string> {
  return {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  };
}

function sseEncode(obj: unknown): Uint8Array {
  const text = `data: ${JSON.stringify(obj)}\n\n`;
  return new TextEncoder().encode(text);
}

function ssePing(): Uint8Array {
  return new TextEncoder().encode(": ping\n\n");
}

async function appKeyFromConfig(env: Env): Promise<string> {
  const cfg = await getFastApiConfig(env);
  const app = (cfg.app ?? {}) as Record<string, unknown>;
  return String(app.app_key ?? "").trim();
}

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

    // D1 `batch([...])` is atomic, but it has a statement-count limit. We keep
    // the single-batch transaction for small token lists, and fall back to
    // multi-batch writes for very large imports.

    const deleteStmt = c.env.grok2api.prepare("DELETE FROM tokens");
    const insertStmts: D1PreparedStatement[] = [];

    const SQLITE_MAX_VARIABLES = 999; // D1/SQLite default
    const PLACEHOLDERS_PER_ROW = 8; // token_type + quota + tags + note ...
    const maxRowsPerStmt = Math.max(1, Math.floor(SQLITE_MAX_VARIABLES / PLACEHOLDERS_PER_ROW));
    const chunkSize = Math.min(120, maxRowsPerStmt); // stay under variable limit

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
      insertStmts.push(c.env.grok2api.prepare(sql).bind(...params));
    }

    const D1_BATCH_LIMIT = 100;
    if (1 + insertStmts.length <= D1_BATCH_LIMIT) {
      await c.env.grok2api.batch([deleteStmt, ...insertStmts]);
    } else {
      await deleteStmt.run();
      for (let i = 0; i < insertStmts.length; i += D1_BATCH_LIMIT) {
        await c.env.grok2api.batch(insertStmts.slice(i, i + D1_BATCH_LIMIT));
      }
    }

    return c.json({ status: "success", message: "Token 已更新" });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

adminFastApiRoutes.post("/tokens/import", requireAppKeyAuth, async (c) => {
  try {
    const body = (await c.req.json()) as unknown;
    if (!isPlainObject(body)) return c.json({ detail: "Invalid payload" }, 400);

    const poolRaw = String(body.pool ?? "").trim();
    const pool = poolRaw === "ssoBasic" || poolRaw === "ssoSuper" ? (poolRaw as "ssoBasic" | "ssoSuper") : null;
    if (!pool) return c.json({ detail: "Invalid pool" }, 400);

    const tokens = parseTokenList(body);
    if (!tokens.length) return c.json({ detail: "No tokens provided" }, 400);

    const tokenType: "sso" | "ssoSuper" = pool === "ssoSuper" ? "ssoSuper" : "sso";
    const defaultQuota = defaultQuotaForPool(pool);
    const quotaRaw = Number(body.quota ?? defaultQuota);
    const quota = Number.isFinite(quotaRaw) ? Math.max(0, Math.floor(quotaRaw)) : defaultQuota;

    const SQLITE_MAX_VARIABLES = 999;
    const PLACEHOLDERS_PER_ROW = 4;
    const chunkSize = Math.max(1, Math.floor(SQLITE_MAX_VARIABLES / PLACEHOLDERS_PER_ROW));

    const now = nowMs();
    let inserted = 0;
    for (let i = 0; i < tokens.length; i += chunkSize) {
      const chunk = tokens.slice(i, i + chunkSize);
      const valuesSql = chunk.map(() => "(?,?,?,?)").join(",");
      const sql = `INSERT OR IGNORE INTO tokens(token, token_type, created_time, remaining_queries) VALUES ${valuesSql}`;
      const params: unknown[] = [];
      for (let j = 0; j < chunk.length; j += 1) {
        params.push(chunk[j], tokenType, now - (i + j), quota);
      }
      const res = await c.env.grok2api.prepare(sql).bind(...params).run();
      inserted += Number((res as any)?.meta?.changes ?? 0);
    }

    return c.json({
      status: "success",
      total: tokens.length,
      inserted,
      skipped: Math.max(0, tokens.length - inserted),
    });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

// ======================================================================
// Batch (SSE) endpoints (FastAPI compatible)
// ======================================================================

adminFastApiRoutes.get("/batch/:taskId/stream", async (c) => {
  const appKey = await appKeyFromConfig(c.env);
  if (!appKey) return c.json({ detail: "App key is not configured" }, 401);

  const url = new URL(c.req.url);
  const queryKey = String(url.searchParams.get("app_key") ?? "").trim();
  if (!queryKey || queryKey !== appKey) {
    return c.json({ detail: "Invalid authentication token" }, 401);
  }

  const taskId = c.req.param("taskId");
  const task = await getBatchTask(c.env.grok2api, taskId);
  if (!task) return c.json({ detail: "Task not found" }, 404);

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let lastPing = nowMs();

      const emitSnapshot = (t: BatchTask) => {
        controller.enqueue(
          sseEncode({
            type: "snapshot",
            task_id: t.task_id,
            status: t.status,
            total: t.total,
            processed: t.processed,
            ok: t.ok,
            fail: t.fail,
            warning: t.warning ?? null,
          }),
        );
      };

      const emitProgress = (t: BatchTask) => {
        controller.enqueue(
          sseEncode({
            type: "progress",
            task_id: t.task_id,
            total: t.total,
            processed: t.processed,
            ok: t.ok,
            fail: t.fail,
          }),
        );
      };

      const emitDone = (t: BatchTask) => {
        controller.enqueue(
          sseEncode({
            type: "done",
            task_id: t.task_id,
            total: t.total,
            processed: t.processed,
            ok: t.ok,
            fail: t.fail,
            warning: t.warning ?? null,
            result: t.result ?? null,
          }),
        );
      };

      const emitError = (t: BatchTask, error: string) => {
        controller.enqueue(
          sseEncode({
            type: "error",
            task_id: t.task_id,
            total: t.total,
            processed: t.processed,
            ok: t.ok,
            fail: t.fail,
            error,
          }),
        );
      };

      const emitCancelled = (t: BatchTask) => {
        controller.enqueue(
          sseEncode({
            type: "cancelled",
            task_id: t.task_id,
            total: t.total,
            processed: t.processed,
            ok: t.ok,
            fail: t.fail,
          }),
        );
      };

      const maybePing = () => {
        const now = nowMs();
        if (now - lastPing < 15000) return;
        lastPing = now;
        controller.enqueue(ssePing());
      };

      try {
        emitSnapshot(task);

        if (task.status === "done") {
          emitDone(task);
          controller.close();
          return;
        }
        if (task.status === "error") {
          emitError(task, task.error ?? "unknown_error");
          controller.close();
          return;
        }
        if (task.status === "cancelled") {
          emitCancelled(task);
          controller.close();
          return;
        }

        task.status = "running";
        await updateBatchTask(c.env.grok2api, task);

        const settingsBundle = await getSettings(c.env);
        const cf = normalizeCfCookie(settingsBundle.grok.cf_clearance ?? "");
        const cfg = await getFastApiConfig(c.env);
        const nsfwCfg = (cfg.nsfw ?? {}) as Record<string, unknown>;
        const featureKey = String(nsfwCfg.feature_key ?? "always_show_nsfw_content").trim() || "always_show_nsfw_content";
        const applyDelayMs = Math.max(0, Math.floor(Number(nsfwCfg.apply_delay_ms ?? 0) || 0));
        const timeoutSec = Math.max(1, Math.floor(Number(nsfwCfg.timeout ?? 60) || 60));
        const timeoutMs = timeoutSec * 1000;

        if (task.kind === "token_refresh") {
          const results: Record<string, boolean> = {};

          for (const token of task.tokens) {
            maybePing();
            if (await isBatchCancelled(c.env.grok2api, task.task_id)) {
              task.status = "cancelled";
              await updateBatchTask(c.env.grok2api, task);
              emitCancelled(task);
              await deleteBatchTask(c.env.grok2api, task.task_id);
              controller.close();
              return;
            }

            const row = await dbFirst<{ token_type: "sso" | "ssoSuper" }>(
              c.env.grok2api,
              "SELECT token_type FROM tokens WHERE token = ?",
              [token],
            );
            const tokenType = row?.token_type === "ssoSuper" ? "ssoSuper" : "sso";

            maybePing();
            const ok = await refreshOneToken({
              env: c.env,
              token,
              tokenType,
              settings: settingsBundle.grok,
              cf,
            });
            results[token] = ok;

            task.processed += 1;
            if (ok) task.ok += 1;
            else task.fail += 1;
            emitProgress(task);
            await updateBatchTask(c.env.grok2api, task);
          }

          const final = {
            status: "success",
            summary: { total: task.total, ok: task.ok, fail: task.fail },
            results,
          };
          task.status = "done";
          task.result = final;
          await updateBatchTask(c.env.grok2api, task);
          emitDone(task);
          await deleteBatchTask(c.env.grok2api, task.task_id);
          controller.close();
          return;
        }

        if (task.kind === "nsfw_enable") {
          const results: Record<string, unknown> = {};

          for (const token of task.tokens) {
            maybePing();
            if (await isBatchCancelled(c.env.grok2api, task.task_id)) {
              task.status = "cancelled";
              await updateBatchTask(c.env.grok2api, task);
              emitCancelled(task);
              await deleteBatchTask(c.env.grok2api, task.task_id);
              controller.close();
              return;
            }

            const outKey = maskToken(token);
            const cookie = buildCookie(token, cf);
            maybePing();
            const res = await enableNsfw({
              cookie,
              settings: settingsBundle.grok,
              featureKey,
              timeoutMs,
            });
            results[outKey] = res;

            if (res.success) {
              const tokenRow = await dbFirst<{ tags: string }>(c.env.grok2api, "SELECT tags FROM tokens WHERE token = ?", [
                token,
              ]);
              const tags = parseTags(tokenRow?.tags ?? "[]");
              if (!tags.includes("nsfw")) tags.push("nsfw");
              await dbRun(c.env.grok2api, "UPDATE tokens SET tags = ? WHERE token = ?", [JSON.stringify(tags), token]);
              if (applyDelayMs > 0) {
                maybePing();
                await new Promise((r) => setTimeout(r, applyDelayMs));
              }
            }

            task.processed += 1;
            if (res.success) task.ok += 1;
            else task.fail += 1;
            emitProgress(task);
            await updateBatchTask(c.env.grok2api, task);
          }

          const final = {
            status: "success",
            summary: { total: task.total, ok: task.ok, fail: task.fail },
            results,
          };
          task.status = "done";
          task.result = final;
          await updateBatchTask(c.env.grok2api, task);
          emitDone(task);
          await deleteBatchTask(c.env.grok2api, task.task_id);
          controller.close();
          return;
        }

        task.status = "error";
        task.error = `Unknown task kind: ${task.kind}`;
        await updateBatchTask(c.env.grok2api, task);
        emitError(task, task.error);
        await deleteBatchTask(c.env.grok2api, task.task_id);
        controller.close();
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        task.status = "error";
        task.error = msg;
        await updateBatchTask(c.env.grok2api, task);
        emitError(task, msg);
        await deleteBatchTask(c.env.grok2api, task.task_id);
        controller.close();
      }
    },
  });

  return new Response(stream, { status: 200, headers: sseHeaders() });
});

adminFastApiRoutes.post("/batch/:taskId/cancel", requireAppKeyAuth, async (c) => {
  const taskId = c.req.param("taskId");
  await markBatchCancelled(c.env.grok2api, taskId);
  return c.json({ status: "success" });
});

// ======================================================================
// Tokens refresh / NSFW enable
// ======================================================================

adminFastApiRoutes.post("/tokens/refresh", requireAppKeyAuth, async (c) => {
  try {
    const body = (await c.req.json()) as unknown;
    const tokens = parseTokenList(body);
    if (!tokens.length) return c.json({ detail: "No tokens provided" }, 400);

    const settingsBundle = await getSettings(c.env);
    const cf = normalizeCfCookie(settingsBundle.grok.cf_clearance ?? "");
    const results: Record<string, boolean> = {};
    for (const token of tokens) {
      const row = await dbFirst<{ token_type: "sso" | "ssoSuper" }>(
        c.env.grok2api,
        "SELECT token_type FROM tokens WHERE token = ?",
        [token],
      );
      const tokenType = row?.token_type === "ssoSuper" ? "ssoSuper" : "sso";
      results[token] = await refreshOneToken({
        env: c.env,
        token,
        tokenType,
        settings: settingsBundle.grok,
        cf,
      });
    }
    return c.json({ status: "success", results });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

adminFastApiRoutes.post("/tokens/refresh/async", requireAppKeyAuth, async (c) => {
  try {
    const body = (await c.req.json()) as unknown;
    const tokens = parseTokenList(body);
    if (!tokens.length) return c.json({ detail: "No tokens provided" }, 400);

    const task = await createBatchTask(c.env.grok2api, { kind: "token_refresh", tokens });
    return c.json({ status: "success", task_id: task.task_id, total: task.total });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

adminFastApiRoutes.post("/tokens/nsfw/enable", requireAppKeyAuth, async (c) => {
  try {
    const body = (await c.req.json()) as unknown;
    let tokens = parseTokenList(body);
    if (!tokens.length) {
      const rows = await dbAll<{ token: string }>(c.env.grok2api, "SELECT token FROM tokens ORDER BY created_time DESC");
      tokens = rows.map((r) => r.token).filter(Boolean);
    }
    if (!tokens.length) return c.json({ detail: "No tokens available" }, 400);

    const settingsBundle = await getSettings(c.env);
    const cf = normalizeCfCookie(settingsBundle.grok.cf_clearance ?? "");
    const cfg = await getFastApiConfig(c.env);
    const nsfwCfg = (cfg.nsfw ?? {}) as Record<string, unknown>;
    const featureKey = String(nsfwCfg.feature_key ?? "always_show_nsfw_content").trim() || "always_show_nsfw_content";
    const applyDelayMs = Math.max(0, Math.floor(Number(nsfwCfg.apply_delay_ms ?? 0) || 0));
    const timeoutSec = Math.max(1, Math.floor(Number(nsfwCfg.timeout ?? 60) || 60));
    const timeoutMs = timeoutSec * 1000;

    const results: Record<string, unknown> = {};
    let ok = 0;
    let fail = 0;
    for (const token of tokens) {
      const cookie = buildCookie(token, cf);
      const res = await enableNsfw({
        cookie,
        settings: settingsBundle.grok,
        featureKey,
        timeoutMs,
      });
      results[maskToken(token)] = res;
      if (res.success) {
        ok += 1;
        const tokenRow = await dbFirst<{ tags: string }>(c.env.grok2api, "SELECT tags FROM tokens WHERE token = ?", [
          token,
        ]);
        const tags = parseTags(tokenRow?.tags ?? "[]");
        if (!tags.includes("nsfw")) tags.push("nsfw");
        await dbRun(c.env.grok2api, "UPDATE tokens SET tags = ? WHERE token = ?", [JSON.stringify(tags), token]);
        if (applyDelayMs > 0) {
          await new Promise((r) => setTimeout(r, applyDelayMs));
        }
      } else {
        fail += 1;
      }
    }

    return c.json({
      status: "success",
      summary: { total: tokens.length, ok, fail },
      results,
    });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

adminFastApiRoutes.post("/tokens/nsfw/enable/async", requireAppKeyAuth, async (c) => {
  try {
    const body = (await c.req.json()) as unknown;
    let tokens = parseTokenList(body);
    if (!tokens.length) {
      const rows = await dbAll<{ token: string }>(c.env.grok2api, "SELECT token FROM tokens ORDER BY created_time DESC");
      tokens = rows.map((r) => r.token).filter(Boolean);
    }
    if (!tokens.length) return c.json({ detail: "No tokens available" }, 400);

    const task = await createBatchTask(c.env.grok2api, { kind: "nsfw_enable", tokens });
    return c.json({ status: "success", task_id: task.task_id, total: task.total });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

// ======================================================================
// Cache endpoints (Workers KV + D1 metadata)
// ======================================================================

function parseCacheType(input: string | null): "image" | "video" {
  const t = String(input ?? "").trim().toLowerCase();
  return t === "video" ? "video" : "image";
}

function mb(bytes: number): number {
  return Math.round((bytes / 1024 / 1024) * 100) / 100;
}

async function cacheCount(db: Env["grok2api"], type: "image" | "video"): Promise<number> {
  const row = await dbFirst<{ c: number }>(db, "SELECT COUNT(1) as c FROM kv_cache WHERE type = ?", [type]);
  return row?.c ?? 0;
}

adminFastApiRoutes.get("/cache", requireAppKeyAuth, async (c) => {
  try {
    const bytes = await getCacheSizeBytes(c.env.grok2api);
    const imageCount = await cacheCount(c.env.grok2api, "image");
    const videoCount = await cacheCount(c.env.grok2api, "video");

    const rows = await dbAll<{ token: string; token_type: "sso" | "ssoSuper"; status: string }>(
      c.env.grok2api,
      "SELECT token, token_type, status FROM tokens ORDER BY created_time DESC",
    );
    const accounts = rows.map((r) => ({
      token: r.token,
      token_masked: r.token.length > 24 ? `${r.token.slice(0, 8)}...${r.token.slice(-16)}` : r.token,
      pool: r.token_type === "ssoSuper" ? "ssoSuper" : "ssoBasic",
      status: normalizeStatus(r.status),
      last_asset_clear_at: null,
    }));

    return c.json({
      local_image: { count: imageCount, size_mb: mb(bytes.image) },
      local_video: { count: videoCount, size_mb: mb(bytes.video) },
      online: { count: 0, status: "not_loaded", token: null, last_asset_clear_at: null },
      online_accounts: accounts,
      online_scope: "none",
      online_details: [],
    });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

adminFastApiRoutes.get("/cache/list", requireAppKeyAuth, async (c) => {
  try {
    const url = new URL(c.req.url);
    const type = parseCacheType(url.searchParams.get("type"));
    const pageRaw = Number(url.searchParams.get("page") ?? 1);
    const pageSizeRaw = Number(url.searchParams.get("page_size") ?? 1000);
    const page = Number.isFinite(pageRaw) ? Math.max(1, Math.floor(pageRaw)) : 1;
    const page_size = Number.isFinite(pageSizeRaw) ? Math.max(1, Math.floor(pageSizeRaw)) : 1000;
    const offset = (page - 1) * page_size;

    const { total, items } = await listCacheRowsByType(c.env.grok2api, type, page_size, offset);
    const mapped = items.map((row) => {
      const prefix = `${type}/`;
      const name = row.key.startsWith(prefix) ? row.key.slice(prefix.length) : row.key;
      const preview_url = type === "image" ? `/images/${encodeURIComponent(name)}` : undefined;
      return {
        name,
        size_bytes: row.size,
        mtime_ms: row.last_access_at || row.created_at,
        ...(preview_url ? { preview_url } : {}),
        view_url: `/v1/files/${type}/${encodeURIComponent(name)}`,
      };
    });

    return c.json({ status: "success", total, page, page_size, items: mapped });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

adminFastApiRoutes.post("/cache/item/delete", requireAppKeyAuth, async (c) => {
  try {
    const body = (await c.req.json()) as unknown;
    if (!isPlainObject(body)) return c.json({ detail: "Invalid payload" }, 400);

    const type = parseCacheType(typeof body.type === "string" ? body.type : null);
    const name = String(body.name ?? "").trim();
    if (!name) return c.json({ detail: "Missing file name" }, 400);

    const key = `${type}/${name}`;
    await c.env.grok2api_cache.delete(key);
    await deleteCacheRow(c.env.grok2api, key);

    return c.json({ status: "success", result: { deleted: true } });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

adminFastApiRoutes.post("/cache/clear", requireAppKeyAuth, async (c) => {
  try {
    const body = (await c.req.json()) as unknown;
    if (!isPlainObject(body)) return c.json({ detail: "Invalid payload" }, 400);

    const type = parseCacheType(typeof body.type === "string" ? body.type : null);

    let count = 0;
    let freed = 0;
    const batch = 500;

    for (let i = 0; i < 500; i++) {
      const rows = await listOldestRows(c.env.grok2api, type, null, batch);
      if (!rows.length) break;
      const keys = rows.map((r) => r.key);
      freed += rows.reduce((sum, r) => sum + (Number(r.size) || 0), 0);
      await Promise.all(keys.map((k) => c.env.grok2api_cache.delete(k)));
      await deleteCacheRows(c.env.grok2api, keys);
      count += keys.length;
      if (keys.length < batch) break;
    }

    return c.json({ status: "success", result: { count, size_mb: mb(freed) } });
  } catch (e) {
    return c.json({ detail: e instanceof Error ? e.message : String(e) }, 500);
  }
});

// Online assets management (Grok account assets) is not supported in Workers build yet.
adminFastApiRoutes.all("/cache/online/load/async", requireAppKeyAuth, (c) =>
  c.json({ detail: "Online asset management is not implemented in Cloudflare Workers build" }, 501),
);
adminFastApiRoutes.all("/cache/online/clear/async", requireAppKeyAuth, (c) =>
  c.json({ detail: "Online asset management is not implemented in Cloudflare Workers build" }, 501),
);
adminFastApiRoutes.all("/cache/online/clear", requireAppKeyAuth, (c) =>
  c.json({ detail: "Online asset management is not implemented in Cloudflare Workers build" }, 501),
);
