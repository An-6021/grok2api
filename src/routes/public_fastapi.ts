import { Hono } from "hono";
import type { Env } from "../env";
import { getFastApiConfig } from "../fastapiConfig";

function bearerToken(authHeader: string | null): string | null {
  if (!authHeader) return null;
  const m = authHeader.match(/^Bearer\s+(.+)$/i);
  return m?.[1]?.trim() || null;
}

export const publicFastApiRoutes = new Hono<{ Bindings: Env }>();

publicFastApiRoutes.get("/verify", async (c) => {
  const cfg = await getFastApiConfig(c.env);
  const app = (cfg.app ?? {}) as Record<string, unknown>;

  const enabled = Boolean(app.public_enabled);
  const publicKey = String(app.public_key ?? "").trim();

  if (!enabled) {
    return c.json({ detail: "Public access is disabled" }, 401);
  }

  // If enabled but no password, allow anonymous access.
  if (!publicKey) {
    return c.json({ status: "success" });
  }

  const token = bearerToken(c.req.header("Authorization") ?? null);
  if (!token) {
    return c.json({ detail: "Missing authentication token" }, 401, {
      "WWW-Authenticate": "Bearer",
    });
  }

  if (token !== publicKey) {
    return c.json({ detail: "Invalid authentication token" }, 401, {
      "WWW-Authenticate": "Bearer",
    });
  }

  return c.json({ status: "success" });
});

publicFastApiRoutes.get("/imagine/config", async (c) => {
  const cfg = await getFastApiConfig(c.env);
  const imageCfg = (cfg.image ?? {}) as Record<string, unknown>;

  const toInt = (raw: unknown, fallback: number) => {
    const n = Number(raw);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(0, Math.floor(n));
  };

  return c.json({
    final_min_bytes: toInt(imageCfg.final_min_bytes, 0),
    medium_min_bytes: toInt(imageCfg.medium_min_bytes, 0),
    nsfw: Boolean(imageCfg.nsfw),
  });
});

// Phase-1: other /v1/public endpoints are intentionally not implemented.
publicFastApiRoutes.all("/*", () => {
  return new Response(
    JSON.stringify({ status: "error", error: "Not implemented in Cloudflare Workers build" }),
    { status: 501, headers: { "content-type": "application/json; charset=utf-8" } },
  );
});
