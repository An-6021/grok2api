import type { Env } from "./env";
import { getFastApiConfig } from "./fastapiConfig";

export interface GlobalSettings {
  base_url?: string;
  log_level?: string;
  image_mode?: "url" | "base64" | "b64_json";
  admin_username?: string;
  admin_password?: string;
  image_cache_max_size_mb?: number;
  video_cache_max_size_mb?: number;
}

export interface GrokSettings {
  api_key?: string;
  proxy_url?: string;
  proxy_pool_url?: string;
  proxy_pool_interval?: number;
  cache_proxy_url?: string;
  cf_clearance?: string; // stored as VALUE only (no "cf_clearance=" prefix)
  x_statsig_id?: string;
  dynamic_statsig?: boolean;
  filtered_tags?: string;
  show_thinking?: boolean;
  custom_personality_default?: string;
  temporary?: boolean;
  video_poster_preview?: boolean;
  stream_first_response_timeout?: number;
  stream_chunk_timeout?: number;
  stream_total_timeout?: number;
  retry_status_codes?: number[];
  image_generation_method?: string;
}

export interface TokenSettings {
  auto_refresh?: boolean;
  refresh_interval_hours?: number;
  fail_threshold?: number;
  save_delay_ms?: number;
  reload_interval_sec?: number;
}

export interface CacheSettings {
  enable_auto_clean?: boolean;
  limit_mb?: number;
  keep_base64_cache?: boolean;
}

export interface PerformanceSettings {
  assets_max_concurrent?: number;
  media_max_concurrent?: number;
  usage_max_concurrent?: number;
  assets_delete_batch_size?: number;
  admin_assets_batch_size?: number;
}

export interface RegisterSettings {
  worker_domain?: string;
  email_domain?: string;
  admin_password?: string;
  yescaptcha_key?: string;
  solver_url?: string;
  solver_browser_type?: string;
  solver_threads?: number;
  register_threads?: number;
  default_count?: number;
  auto_start_solver?: boolean;
  solver_debug?: boolean;
  max_errors?: number;
  max_runtime_minutes?: number;
}

export interface SettingsBundle {
  global: Required<GlobalSettings>;
  grok: Required<GrokSettings>;
  token: Required<TokenSettings>;
  cache: Required<CacheSettings>;
  performance: Required<PerformanceSettings>;
  register: Required<RegisterSettings>;
}

const DEFAULTS: SettingsBundle = {
  global: {
    base_url: "",
    log_level: "INFO",
    image_mode: "url",
    admin_username: "admin",
    admin_password: "admin",
    image_cache_max_size_mb: 512,
    video_cache_max_size_mb: 1024,
  },
  grok: {
    api_key: "",
    proxy_url: "",
    proxy_pool_url: "",
    proxy_pool_interval: 300,
    cache_proxy_url: "",
    cf_clearance: "",
    x_statsig_id: "",
    dynamic_statsig: true,
    filtered_tags: "xaiartifact,xai:tool_usage_card",
    show_thinking: true,
    custom_personality_default: "",
    temporary: false,
    video_poster_preview: false,
    stream_first_response_timeout: 30,
    stream_chunk_timeout: 120,
    stream_total_timeout: 600,
    retry_status_codes: [401, 429, 403],
    image_generation_method: "legacy",
  },
  token: {
    auto_refresh: true,
    refresh_interval_hours: 8,
    fail_threshold: 5,
    save_delay_ms: 500,
    reload_interval_sec: 30,
  },
  cache: {
    enable_auto_clean: true,
    limit_mb: 1024,
    keep_base64_cache: true,
  },
  performance: {
    assets_max_concurrent: 25,
    media_max_concurrent: 50,
    usage_max_concurrent: 25,
    assets_delete_batch_size: 10,
    admin_assets_batch_size: 10,
  },
  register: {
    worker_domain: "",
    email_domain: "",
    admin_password: "",
    yescaptcha_key: "",
    solver_url: "http://127.0.0.1:5072",
    solver_browser_type: "camoufox",
    solver_threads: 5,
    register_threads: 10,
    default_count: 100,
    auto_start_solver: true,
    solver_debug: false,
    max_errors: 0,
    max_runtime_minutes: 0,
  },
};

const IMAGE_METHOD_LEGACY = "legacy";
const IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL = "imagine_ws_experimental";
const IMAGE_METHOD_ALIASES: Record<string, string> = {
  imagine_ws: IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL,
  experimental: IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL,
  new: IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL,
  new_method: IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL,
};

function stripCfPrefix(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "";
  return trimmed.startsWith("cf_clearance=")
    ? trimmed.slice("cf_clearance=".length)
    : trimmed;
}

export function normalizeCfCookie(value: string): string {
  const cleaned = stripCfPrefix(value);
  return cleaned ? `cf_clearance=${cleaned}` : "";
}

export function normalizeImageGenerationMethod(value: unknown): string {
  const candidate = String(value ?? "")
    .trim()
    .toLowerCase();
  if (candidate === IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL) {
    return IMAGE_METHOD_IMAGINE_WS_EXPERIMENTAL;
  }
  if (IMAGE_METHOD_ALIASES[candidate]) {
    return IMAGE_METHOD_ALIASES[candidate];
  }
  return IMAGE_METHOD_LEGACY;
}

function asNumber(v: unknown, fallback: number): number {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return n;
}

function asBool(v: unknown, fallback: boolean): boolean {
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v === 1;
  if (typeof v !== "string") return fallback;
  const s = v.trim().toLowerCase();
  if (s === "true" || s === "1" || s === "yes") return true;
  if (s === "false" || s === "0" || s === "no") return false;
  return fallback;
}

function asString(v: unknown, fallback: string): string {
  const s = String(v ?? "").trim();
  return s ? s : fallback;
}

function joinFilterTags(v: unknown, fallback: string): string {
  if (Array.isArray(v)) {
    return v
      .map((x) => String(x ?? "").trim())
      .filter(Boolean)
      .join(",");
  }
  return asString(v, fallback);
}

export async function getSettings(env: Env): Promise<SettingsBundle> {
  const cfg = await getFastApiConfig(env);

  const app = (cfg.app ?? {}) as Record<string, unknown>;
  const proxy = (cfg.proxy ?? {}) as Record<string, unknown>;
  const retry = (cfg.retry ?? {}) as Record<string, unknown>;
  const token = (cfg.token ?? {}) as Record<string, unknown>;
  const cache = (cfg.cache ?? {}) as Record<string, unknown>;
  const chat = (cfg.chat ?? {}) as Record<string, unknown>;
  const video = (cfg.video ?? {}) as Record<string, unknown>;
  const usage = (cfg.usage ?? {}) as Record<string, unknown>;
  const asset = (cfg.asset ?? {}) as Record<string, unknown>;

  const limitMb = asNumber(cache.limit_mb, DEFAULTS.cache.limit_mb);
  const mediaMaxConcurrent = Math.max(
    1,
    Math.floor(
      Math.max(
        asNumber(chat.concurrent, DEFAULTS.performance.media_max_concurrent),
        asNumber(video.concurrent, DEFAULTS.performance.media_max_concurrent),
      ),
    ),
  );
  const assetsMaxConcurrent = Math.max(
    1,
    Math.floor(asNumber(asset.upload_concurrent, DEFAULTS.performance.assets_max_concurrent)),
  );

  const bundle: SettingsBundle = {
    global: {
      ...DEFAULTS.global,
      base_url: asString(app.app_url, DEFAULTS.global.base_url),
      // Prefer config-provided admin password (FastAPI "app_key").
      admin_password: asString(app.app_key, DEFAULTS.global.admin_password),
      // Prefer image format from config ("url"|"base64"), but keep compatible union.
      image_mode: (String(app.image_format ?? DEFAULTS.global.image_mode).trim().toLowerCase() as
        | "url"
        | "base64"
        | "b64_json") ?? DEFAULTS.global.image_mode,
      image_cache_max_size_mb: limitMb,
      video_cache_max_size_mb: limitMb,
    },
    grok: {
      ...DEFAULTS.grok,
      api_key: asString(app.api_key, DEFAULTS.grok.api_key),
      proxy_url: asString(proxy.base_proxy_url, DEFAULTS.grok.proxy_url),
      cache_proxy_url: asString(proxy.asset_proxy_url, DEFAULTS.grok.cache_proxy_url),
      cf_clearance: stripCfPrefix(asString(proxy.cf_clearance, DEFAULTS.grok.cf_clearance)),
      dynamic_statsig: asBool(app.dynamic_statsig, DEFAULTS.grok.dynamic_statsig),
      filtered_tags: joinFilterTags(app.filter_tags, DEFAULTS.grok.filtered_tags),
      show_thinking: asBool(app.thinking, DEFAULTS.grok.show_thinking),
      custom_personality_default: asString(
        (app as any).custom_personality_default,
        DEFAULTS.grok.custom_personality_default,
      ),
      temporary: asBool(app.temporary, DEFAULTS.grok.temporary),
      stream_chunk_timeout: asNumber(chat.stream_timeout, DEFAULTS.grok.stream_chunk_timeout),
      stream_total_timeout: asNumber(chat.timeout, DEFAULTS.grok.stream_total_timeout),
      retry_status_codes: Array.isArray(retry.retry_status_codes)
        ? (retry.retry_status_codes as unknown[])
            .map((n) => Number(n))
            .filter((n) => Number.isFinite(n))
            .map((n) => Math.floor(n))
        : DEFAULTS.grok.retry_status_codes,
      image_generation_method: normalizeImageGenerationMethod(
        (cfg as any)?.image?.generation_method ?? DEFAULTS.grok.image_generation_method,
      ),
    },
    token: {
      ...DEFAULTS.token,
      auto_refresh: asBool(token.auto_refresh, DEFAULTS.token.auto_refresh),
      refresh_interval_hours: asNumber(token.refresh_interval_hours, DEFAULTS.token.refresh_interval_hours),
      fail_threshold: asNumber(token.fail_threshold, DEFAULTS.token.fail_threshold),
      save_delay_ms: asNumber(token.save_delay_ms, DEFAULTS.token.save_delay_ms),
      reload_interval_sec: asNumber(token.reload_interval_sec, DEFAULTS.token.reload_interval_sec),
    },
    cache: {
      ...DEFAULTS.cache,
      enable_auto_clean: asBool(cache.enable_auto_clean, DEFAULTS.cache.enable_auto_clean),
      limit_mb: limitMb,
      keep_base64_cache: true,
    },
    performance: {
      ...DEFAULTS.performance,
      assets_max_concurrent: assetsMaxConcurrent,
      media_max_concurrent: mediaMaxConcurrent,
      usage_max_concurrent: Math.max(
        1,
        Math.floor(asNumber(usage.concurrent, DEFAULTS.performance.usage_max_concurrent)),
      ),
      assets_delete_batch_size: Math.max(
        1,
        Math.floor(asNumber(asset.delete_batch_size, DEFAULTS.performance.assets_delete_batch_size)),
      ),
      admin_assets_batch_size: Math.max(
        1,
        Math.floor(asNumber(asset.delete_batch_size, DEFAULTS.performance.admin_assets_batch_size)),
      ),
    },
    register: { ...DEFAULTS.register },
  };

  // Normalize/ensure required fields.
  bundle.grok.cf_clearance = stripCfPrefix(bundle.grok.cf_clearance ?? "");
  bundle.grok.image_generation_method = normalizeImageGenerationMethod(bundle.grok.image_generation_method);
  return bundle;
}
