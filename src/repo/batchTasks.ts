import type { Env } from "../env";
import { dbFirst, dbRun } from "../db";
import { nowMs } from "../utils/time";

export type BatchTaskKind = "token_refresh" | "nsfw_enable";
export type BatchTaskStatus = "queued" | "running" | "done" | "error" | "cancelled";

export interface BatchTask {
  task_id: string;
  kind: BatchTaskKind;
  status: BatchTaskStatus;
  total: number;
  processed: number;
  ok: number;
  fail: number;
  tokens: string[];
  warning?: string | null;
  result?: unknown;
  error?: string | null;
  created_at: number;
  updated_at: number;
}

function taskKey(taskId: string): string {
  return `batch_task:${taskId}`;
}

function cancelKey(taskId: string): string {
  return `batch_cancel:${taskId}`;
}

function safeParseJson(raw: string): unknown {
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
}

function normalizeTask(raw: unknown): BatchTask | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;

  const task_id = String(obj.task_id ?? "").trim();
  const kind = String(obj.kind ?? "").trim() as BatchTaskKind;
  const status = String(obj.status ?? "").trim() as BatchTaskStatus;
  const total = Number(obj.total ?? 0);
  const processed = Number(obj.processed ?? 0);
  const ok = Number(obj.ok ?? 0);
  const fail = Number(obj.fail ?? 0);
  const created_at = Number(obj.created_at ?? 0);
  const updated_at = Number(obj.updated_at ?? 0);

  const tokensRaw = obj.tokens;
  const tokens = Array.isArray(tokensRaw)
    ? tokensRaw.map((t) => String(t ?? "").trim()).filter(Boolean)
    : [];

  if (!task_id || !kind || !status) return null;
  if (!Number.isFinite(total) || !Number.isFinite(processed) || !Number.isFinite(ok) || !Number.isFinite(fail))
    return null;

  return {
    task_id,
    kind,
    status,
    total: Math.max(0, Math.floor(total)),
    processed: Math.max(0, Math.floor(processed)),
    ok: Math.max(0, Math.floor(ok)),
    fail: Math.max(0, Math.floor(fail)),
    tokens,
    warning: obj.warning ? String(obj.warning) : null,
    result: obj.result,
    error: obj.error ? String(obj.error) : null,
    created_at: Number.isFinite(created_at) ? Math.floor(created_at) : 0,
    updated_at: Number.isFinite(updated_at) ? Math.floor(updated_at) : 0,
  };
}

export async function createBatchTask(
  db: Env["grok2api"],
  args: { kind: BatchTaskKind; tokens: string[] },
): Promise<BatchTask> {
  const task_id = crypto.randomUUID().replaceAll("-", "");
  const now = nowMs();
  const total = Math.max(0, args.tokens.length);
  const task: BatchTask = {
    task_id,
    kind: args.kind,
    status: "queued",
    total,
    processed: 0,
    ok: 0,
    fail: 0,
    tokens: args.tokens,
    created_at: now,
    updated_at: now,
  };
  await dbRun(db, "INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES(?,?,?)", [
    taskKey(task_id),
    JSON.stringify(task),
    now,
  ]);
  return task;
}

export async function getBatchTask(db: Env["grok2api"], taskId: string): Promise<BatchTask | null> {
  const row = await dbFirst<{ value: string }>(db, "SELECT value FROM settings WHERE key = ?", [taskKey(taskId)]);
  const parsed = row?.value ? safeParseJson(row.value) : null;
  return normalizeTask(parsed);
}

export async function updateBatchTask(db: Env["grok2api"], task: BatchTask): Promise<void> {
  const now = nowMs();
  const next = { ...task, updated_at: now };
  await dbRun(db, "INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES(?,?,?)", [
    taskKey(task.task_id),
    JSON.stringify(next),
    now,
  ]);
}

export async function markBatchCancelled(db: Env["grok2api"], taskId: string): Promise<void> {
  await dbRun(db, "INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES(?,?,?)", [
    cancelKey(taskId),
    "1",
    nowMs(),
  ]);
}

export async function isBatchCancelled(db: Env["grok2api"], taskId: string): Promise<boolean> {
  const row = await dbFirst<{ value: string }>(db, "SELECT value FROM settings WHERE key = ?", [cancelKey(taskId)]);
  return Boolean(row?.value);
}

export async function deleteBatchTask(db: Env["grok2api"], taskId: string): Promise<void> {
  await dbRun(db, "DELETE FROM settings WHERE key IN (?,?)", [taskKey(taskId), cancelKey(taskId)]);
}

