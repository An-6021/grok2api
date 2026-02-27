import type { MiddlewareHandler } from "hono";
import type { Env } from "./env";
import { getFastApiConfig } from "./fastapiConfig";

export interface ApiAuthInfo {
  key: string | null;
  name: string;
  is_admin: boolean;
}

function bearerToken(authHeader: string | null): string | null {
  if (!authHeader) return null;
  const m = authHeader.match(/^Bearer\s+(.+)$/i);
  return m?.[1]?.trim() || null;
}

function authError(message: string, code: string): Record<string, unknown> {
  return {
    error: {
      message,
      type: "authentication_error",
      code,
    },
  };
}

export const requireApiAuth: MiddlewareHandler<{
  Bindings: Env;
  Variables: { apiAuth: ApiAuthInfo };
}> = async (c, next) => {
  if (c.req.method === "OPTIONS") {
    c.set("apiAuth", { key: null, name: "Anonymous", is_admin: false });
    return next();
  }

  const token = bearerToken(c.req.header("Authorization") ?? null);
  const cfg = await getFastApiConfig(c.env);
  const app = (cfg.app ?? {}) as Record<string, unknown>;
  const globalKey = String(app.api_key ?? "").trim();

  // FastAPI-compatible: if api_key is empty, auth is disabled.
  if (!globalKey) {
    c.set("apiAuth", { key: null, name: "Anonymous", is_admin: false });
    return next();
  }

  if (!token) {
    return c.json(authError("Missing authentication token", "missing_token"), 401, {
      "WWW-Authenticate": "Bearer",
    });
  }

  if (token !== globalKey) {
    return c.json(authError("Invalid authentication token", "invalid_token"), 401, {
      "WWW-Authenticate": "Bearer",
    });
  }

  c.set("apiAuth", { key: token, name: "Admin", is_admin: true });
  return next();
};

export const requireAppKeyAuth: MiddlewareHandler<{ Bindings: Env }> = async (
  c,
  next,
) => {
  if (c.req.method === "OPTIONS") return next();

  const token = bearerToken(c.req.header("Authorization") ?? null);
  const cfg = await getFastApiConfig(c.env);
  const app = (cfg.app ?? {}) as Record<string, unknown>;
  const appKey = String(app.app_key ?? "").trim();

  if (!appKey) {
    return c.json(
      { detail: "App key is not configured" },
      401,
      { "WWW-Authenticate": "Bearer" },
    );
  }

  if (!token) {
    return c.json(
      { detail: "Missing authentication token" },
      401,
      { "WWW-Authenticate": "Bearer" },
    );
  }

  if (token !== appKey) {
    return c.json(
      { detail: "Invalid authentication token" },
      401,
      { "WWW-Authenticate": "Bearer" },
    );
  }

  return next();
};

