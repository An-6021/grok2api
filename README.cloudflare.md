# Grok2API (Cloudflare Workers / Pages: D1 + KV)

本仓库提供 **Cloudflare Workers** 可部署版本（TypeScript），与当前 FastAPI 版保持接口语义尽量一致：

- API: `/v1/*`（OpenAI 兼容）
- 管理后台: `/admin/*` + `/v1/admin/*`
- 静态资源: `/static/*`（来自 `app/static`）
- 媒体代理与缓存: `/images/*`（KV 缓存 + D1 记录）

> 说明：一期仅保证后台“核心能力”可用（登录验证、Token 管理、配置读写、存储类型）。`/v1/public/*`、`/v1/admin/cache*`、`/v1/admin/tokens/refresh*` 等暂不实现（返回 501）。

## 1) 手动部署（本地）

前置：

- Node.js + npm
- Cloudflare 账号（Workers、D1、KV）

安装依赖：

```bash
npm ci
```

首次部署（交互式：自动创建/复用 D1 & KV，并写回 `wrangler.toml`，然后 migrations + deploy）：

```bash
./scripts/cf_deploy.sh
```

> 如你想手动创建资源，也可以先执行：
>
> - `npx wrangler d1 create grok2api`
> - `npx wrangler kv namespace create grok2api-cache`

## 2) GitHub Actions 一键部署（推荐）

工作流文件：`.github/workflows/cloudflare-workers.yml`

在 GitHub 仓库设置 Secrets（Settings → Secrets and variables → Actions）：

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

随后 push 到 `main`，Actions 会：

1. 自动创建/复用 D1 数据库与 KV Namespace
2. 生成 `wrangler.ci.toml`（填入 D1/KV 的 ID，并注入 `BUILD_SHA`）
3. 远端应用 migrations
4. 部署 Worker

> API Token 建议使用 Cloudflare **API Token**（不要用 Global API Key），并确保至少包含 Workers Scripts / D1 / Workers KV Storage 的编辑权限。

## 3) 部署后初始化（必须）

1. 打开后台：`https://<你的域名>/admin/login`
2. 输入后台密码（默认 `grok2api`，对应配置 `app.app_key`）
3. 进入 `/admin/token` 添加 Token（`ssoBasic` / `ssoSuper`）
4. 进入 `/admin/config` 按需配置：
   - `app.api_key`：是否开启 `/v1/*` 访问鉴权（为空则不鉴权）
   - `proxy.base_proxy_url` / `proxy.asset_proxy_url`：可选代理
   - `proxy.cf_clearance`：可选（Cloudflare 验证 cookie）

## 4) 健康检查

```bash
curl -s https://<你的域名>/health
```

## 5) 关键路由

- `GET /health`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `GET /images/:imgPath`
- `GET /admin/login`
- `GET /admin/token`
- `GET /admin/config`
