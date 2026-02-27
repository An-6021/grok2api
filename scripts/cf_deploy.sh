#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="wrangler.toml"
MODE="" # first | redeploy
DO_INSTALL=0
SKIP_TYPECHECK=0
SKIP_MIGRATE=0
NO_COLOR="${NO_COLOR:-1}"

usage() {
  cat <<'EOF'
用法:
  scripts/cf_deploy.sh [--mode first|redeploy] [--config <path>] [--install] [--skip-typecheck] [--skip-migrate]

默认行为:
  - 交互式终端中未指定 --mode 时，会弹出菜单让你选择:
      1) 首次部署
      2) 重新部署
  - 默认使用 wrangler.toml（可用 --config 指定其它配置文件）

首次部署会:
  - 确保 D1 / KV 已存在（缺失则创建）
  - 自动把 D1/KV 的 ID 写回配置文件
  - 执行 migrations + 部署

Examples:
  ./scripts/cf_deploy.sh
  ./scripts/cf_deploy.sh --mode first
  ./scripts/cf_deploy.sh --mode redeploy
  ./scripts/cf_deploy.sh --skip-migrate
  ./scripts/cf_deploy.sh --config wrangler.ci.toml
EOF
}

load_wrangler_toml() {
  local file="$1"
  local tmp
  tmp="$(mktemp -t grok2api-wrangler.XXXXXX)"

  # NOTE: macOS ships bash 3.2 which is fragile with `eval "$(cmd <<HEREDOC)"`.
  # Write assignments to a temp file then source it.
  python - "$file" >"$tmp" <<'PY'
import shlex
import sys
import tomllib

path = sys.argv[1]
with open(path, "rb") as f:
    data = tomllib.load(f)

def q(v) -> str:
    return shlex.quote("" if v is None else str(v))

def first_list(v):
    return v[0] if isinstance(v, list) and v else {}

d1 = first_list(data.get("d1_databases"))
kv = first_list(data.get("kv_namespaces"))

print(f"WORKER_NAME={q(data.get('name',''))}")
print(f"D1_BINDING={q(d1.get('binding',''))}")
print(f"D1_NAME={q(d1.get('database_name',''))}")
print(f"D1_ID={q(d1.get('database_id',''))}")
print(f"KV_BINDING={q(kv.get('binding',''))}")
print(f"KV_ID={q(kv.get('id',''))}")
PY

  # shellcheck disable=SC1090
  source "$tmp"
  rm -f "$tmp"
}

is_missing_id() {
  local v="${1:-}"
  [[ -z "$v" ]] && return 0
  [[ "$v" == "REPLACE_WITH_D1_DATABASE_ID" ]] && return 0
  [[ "$v" == "REPLACE_WITH_KV_NAMESPACE_ID" ]] && return 0
  return 1
}

toml_patch_first_resource_id() {
  local file="$1"
  local table="$2" # d1_databases | kv_namespaces
  local key="$3"   # database_id | id
  local value="$4"
  python - "$file" "$table" "$key" "$value" <<'PY'
import re
import sys

path, table, key, value = sys.argv[1:]
text = open(path, "r", encoding="utf-8").read()

block_re = re.compile(rf"(?ms)(^\\[\\[{re.escape(table)}\\]\\].*?)(?=^\\[\\[|\\Z)")
m = block_re.search(text)
if not m:
    raise SystemExit(f"Missing [[{table}]] block in {path}")

block = m.group(1)

line_re = re.compile(rf'(?m)^(\\s*{re.escape(key)}\\s*=\\s*\")[^\"]*(\"\\s*)$')
if line_re.search(block):
    block2 = line_re.sub(rf"\\1{value}\\2", block, count=1)
else:
    # Insert after first line of the block for readability.
    lines = block.splitlines(True)
    if not lines:
        raise SystemExit(f"Empty [[{table}]] block in {path}")
    lines.insert(1, f'{key} = \"{value}\"\\n')
    block2 = "".join(lines)

text2 = text[: m.start(1)] + block2 + text[m.end(1) :]
open(path, "w", encoding="utf-8").write(text2)
print(f"Updated {path}: [[{table}]] {key} = {value}")
PY
}

require_wrangler_login() {
  if NO_COLOR="$NO_COLOR" npx wrangler whoami >/dev/null 2>&1; then
    return 0
  fi
  echo "Wrangler 未登录，请先执行: npx wrangler login" >&2
  exit 2
}

find_d1_uuid_by_name() {
  local name="$1"
  NO_COLOR="$NO_COLOR" npx wrangler d1 list --json | python - "$name" <<'PY'
import json
import sys

name = sys.argv[1]
data = json.load(sys.stdin)
for item in data:
    if item.get("name") == name:
        print(item.get("uuid") or item.get("id") or "")
        break
PY
}

find_kv_id_by_title() {
  local title="$1"
  NO_COLOR="$NO_COLOR" npx wrangler kv namespace list | python - "$title" <<'PY'
import json
import sys

title = sys.argv[1]
data = json.load(sys.stdin)
for item in data:
    if item.get("title") == title:
        print(item.get("id") or "")
        break
PY
}

ensure_resources() {
  local allow_create="$1" # 1 or 0

  require_wrangler_login

  # D1
  if is_missing_id "$D1_ID"; then
    local found
    found="$(find_d1_uuid_by_name "$D1_NAME" | tr -d '\r\n' || true)"
    if [[ -z "$found" && "$allow_create" -eq 1 ]]; then
      echo "创建 D1 数据库: $D1_NAME"
      NO_COLOR="$NO_COLOR" npx wrangler d1 create "$D1_NAME" >/dev/null
      found="$(find_d1_uuid_by_name "$D1_NAME" | tr -d '\r\n' || true)"
    fi
    if [[ -z "$found" ]]; then
      echo "未找到 D1 数据库: $D1_NAME" >&2
      if [[ "$allow_create" -eq 0 ]]; then
        echo "请使用首次部署模式自动创建: ./scripts/cf_deploy.sh --mode first" >&2
      fi
      exit 2
    fi
    toml_patch_first_resource_id "$CONFIG_PATH" "d1_databases" "database_id" "$found"
    D1_ID="$found"
  fi

  # KV
  if is_missing_id "$KV_ID"; then
    local kv_title_default="${WORKER_NAME:-grok2api}-cache"
    local kv_title="$kv_title_default"
    if [[ -t 0 ]]; then
      local input=""
      read -r -p "KV Namespace 名称 [${kv_title_default}]: " input
      if [[ -n "${input// }" ]]; then
        kv_title="$input"
      fi
    fi

    local found
    found="$(find_kv_id_by_title "$kv_title" | tr -d '\r\n' || true)"
    if [[ -z "$found" && "$allow_create" -eq 1 ]]; then
      echo "创建 KV Namespace: $kv_title"
      NO_COLOR="$NO_COLOR" npx wrangler kv namespace create "$kv_title" >/dev/null
      found="$(find_kv_id_by_title "$kv_title" | tr -d '\r\n' || true)"
    fi
    if [[ -z "$found" ]]; then
      echo "未找到 KV Namespace: $kv_title" >&2
      if [[ "$allow_create" -eq 0 ]]; then
        echo "请使用首次部署模式自动创建: ./scripts/cf_deploy.sh --mode first" >&2
      fi
      exit 2
    fi
    toml_patch_first_resource_id "$CONFIG_PATH" "kv_namespaces" "id" "$found"
    KV_ID="$found"
  fi
}

prompt_mode_if_tty() {
  if [[ -n "$MODE" ]]; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    MODE="redeploy"
    return 0
  fi

  echo "========================================"
  echo " Cloudflare Workers 部署助手"
  echo " 配置文件: $CONFIG_PATH"
  echo "========================================"
  echo
  echo "请选择操作:"
  echo "  1) 首次部署 (安装依赖 + 确保 D1/KV + migrations + deploy)"
  echo "  2) 重新部署 (migrations + deploy)"
  echo "  3) 退出"
  echo
  local choice=""
  read -r -p "输入选项 [1-3]: " choice
  case "$choice" in
    1) MODE="first" ;;
    2) MODE="redeploy" ;;
    3) exit 0 ;;
    *) echo "无效选项" >&2; exit 2 ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:-}"
      shift 2
      ;;
    --install)
      DO_INSTALL=1
      shift
      ;;
    --skip-typecheck)
      SKIP_TYPECHECK=1
      shift
      ;;
    --skip-migrate)
      SKIP_MIGRATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG_PATH" ]]; then
  echo "缺少 --config 参数值" >&2
  exit 2
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "配置文件不存在: $CONFIG_PATH" >&2
  exit 2
fi

if [[ -n "$MODE" && "$MODE" != "first" && "$MODE" != "redeploy" ]]; then
  echo "无效 --mode: $MODE (可选: first|redeploy)" >&2
  exit 2
fi

prompt_mode_if_tty

load_wrangler_toml "$CONFIG_PATH"

# This Workers build expects fixed binding names.
if [[ "$D1_BINDING" != "grok2api" ]]; then
  echo "wrangler.toml 里 [[d1_databases]] 的 binding 必须是 \"grok2api\" (代码使用 env.grok2api)，当前是: ${D1_BINDING:-<空>}" >&2
  exit 2
fi
if [[ "$KV_BINDING" != "grok2api_cache" ]]; then
  echo "wrangler.toml 里 [[kv_namespaces]] 的 binding 必须是 \"grok2api_cache\" (代码使用 env.grok2api_cache)，当前是: ${KV_BINDING:-<空>}" >&2
  exit 2
fi
if [[ -z "$D1_NAME" ]]; then
  echo "缺少 [[d1_databases]] database_name: $CONFIG_PATH" >&2
  exit 2
fi

# For `wrangler d1 migrations apply <database>` you can pass either the DB name or the binding.
# Use the DB name when available (this matches "my DB is called grok2api" expectations).
D1_APPLY_TARGET="${D1_NAME:-$D1_BINDING}"
if [[ -z "$D1_APPLY_TARGET" ]]; then
  echo "Could not find D1 binding/database_name in $CONFIG_PATH" >&2
  exit 2
fi

if [[ "$MODE" == "first" ]]; then
  DO_INSTALL=1
fi

echo "项目: ${WORKER_NAME:-<未知>}"
echo "D1: 名称=${D1_NAME} 绑定=${D1_BINDING} ID=${D1_ID:-<缺失>}"
echo "KV: 绑定=${KV_BINDING} ID=${KV_ID:-<缺失>}"
echo

if [[ "$MODE" == "first" ]]; then
  ensure_resources 1
else
  ensure_resources 0
fi

if [[ "$DO_INSTALL" -eq 1 ]]; then
  npm ci
else
  # Smooth DX: if deps are missing, install automatically.
  if [[ ! -d "node_modules" ]]; then
    npm ci
  fi
fi

if [[ "$SKIP_TYPECHECK" -eq 0 ]]; then
  npm run typecheck
fi

if [[ "$SKIP_MIGRATE" -eq 0 ]]; then
  npx wrangler d1 migrations apply "$D1_APPLY_TARGET" --remote --config "$CONFIG_PATH"
fi

npx wrangler deploy --config "$CONFIG_PATH"
