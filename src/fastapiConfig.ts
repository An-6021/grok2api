import type { Env } from "./env";
import { dbFirst, dbRun } from "./db";
import { nowMs } from "./utils/time";

export type FastApiConfig = Record<string, unknown>;

const SETTINGS_KEY = "fastapi_config";

export const DEFAULT_FASTAPI_CONFIG: FastApiConfig = {
  app: {
    app_url: "",
    app_key: "grok2api",
    api_key: "",
    public_enabled: false,
    public_key: "",
    image_format: "url",
    video_format: "html",
    temporary: true,
    disable_memory: true,
    stream: true,
    thinking: true,
    custom_personality_default: "",
    dynamic_statsig: true,
    filter_tags: ["xaiartifact", "xai:tool_usage_card", "grok:render"],
  },
  proxy: {
    base_proxy_url: "",
    asset_proxy_url: "",
    cf_clearance: "",
    browser: "chrome136",
    user_agent:
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
  },
  retry: {
    max_retry: 3,
    retry_status_codes: [401, 429, 403],
    reset_session_status_codes: [403],
    retry_backoff_base: 0.5,
    retry_backoff_factor: 2.0,
    retry_backoff_max: 20.0,
    retry_budget: 60.0,
  },
  token: {
    auto_refresh: true,
    refresh_interval_hours: 8,
    super_refresh_interval_hours: 2,
    fail_threshold: 5,
    save_delay_ms: 500,
    usage_flush_interval_sec: 5,
    reload_interval_sec: 30,
  },
  cache: {
    enable_auto_clean: true,
    limit_mb: 512,
  },
  chat: {
    concurrent: 50,
    timeout: 60,
    stream_timeout: 60,
  },
  image: {
    timeout: 60,
    stream_timeout: 60,
    final_timeout: 15,
    nsfw: true,
    medium_min_bytes: 30000,
    final_min_bytes: 100000,
  },
  video: {
    concurrent: 100,
    timeout: 60,
    stream_timeout: 60,
  },
  voice: {
    timeout: 60,
  },
  asset: {
    upload_concurrent: 100,
    upload_timeout: 60,
    download_concurrent: 100,
    download_timeout: 60,
    list_concurrent: 100,
    list_timeout: 60,
    list_batch_size: 50,
    delete_concurrent: 100,
    delete_timeout: 60,
    delete_batch_size: 50,
  },
  nsfw: {
    concurrent: 60,
    batch_size: 30,
    timeout: 60,
    feature_key: "always_show_nsfw_content",
    apply_delay_ms: 0,
  },
  usage: {
    concurrent: 100,
    batch_size: 50,
    timeout: 60,
  },
};

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function deepMerge(base: unknown, override: unknown): unknown {
  if (!isPlainObject(base)) {
    return isPlainObject(override) ? { ...override } : base;
  }

  const out: Record<string, unknown> = { ...base };
  if (!isPlainObject(override)) return out;

  for (const [k, v] of Object.entries(override)) {
    const existing = out[k];
    if (isPlainObject(existing) && isPlainObject(v)) out[k] = deepMerge(existing, v);
    else out[k] = v;
  }
  return out;
}

function hasAllDefaultKeys(defaults: unknown, candidate: unknown): boolean {
  if (!isPlainObject(defaults)) return true;
  if (!isPlainObject(candidate)) return false;
  for (const [k, v] of Object.entries(defaults)) {
    if (!(k in candidate)) return false;
    if (!hasAllDefaultKeys(v, candidate[k])) return false;
  }
  return true;
}

function safeParseJson(raw: string): unknown {
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
}

function normalizeConfig(raw: unknown): FastApiConfig {
  const merged = deepMerge(DEFAULT_FASTAPI_CONFIG, raw);
  return (isPlainObject(merged) ? merged : { ...DEFAULT_FASTAPI_CONFIG }) as FastApiConfig;
}

export async function getFastApiConfig(env: Env): Promise<FastApiConfig> {
  const row = await dbFirst<{ value: string }>(
    env.grok2api,
    "SELECT value FROM settings WHERE key = ?",
    [SETTINGS_KEY],
  );
  const parsed = row?.value ? safeParseJson(row.value) : null;
  const merged = normalizeConfig(parsed);

  // Backfill missing keys so the admin UI always sees a complete config object.
  const shouldPersist = !row || !hasAllDefaultKeys(DEFAULT_FASTAPI_CONFIG, parsed);
  if (shouldPersist) {
    await dbRun(
      env.grok2api,
      "INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES(?,?,?)",
      [SETTINGS_KEY, JSON.stringify(merged), nowMs()],
    );
  }

  return merged;
}

export async function setFastApiConfig(env: Env, next: unknown): Promise<FastApiConfig> {
  const merged = normalizeConfig(next);

  const app = (merged.app ?? {}) as Record<string, unknown>;
  const appKey = String(app.app_key ?? "").trim();
  if (!appKey) {
    throw new Error("app_key 不能为空（后台密码）");
  }

  await dbRun(
    env.grok2api,
    "INSERT OR REPLACE INTO settings(key, value, updated_at) VALUES(?,?,?)",
    [SETTINGS_KEY, JSON.stringify(merged), nowMs()],
  );

  return merged;
}

