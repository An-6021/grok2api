"""
配置管理

- config.toml: 运行时配置
- config.defaults.toml: 默认配置基线
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
import os
import json
import tomllib

from app.core.logger import logger

DEFAULT_CONFIG_FILE = Path(__file__).parent.parent.parent / "config.defaults.toml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并字典: override 覆盖 base."""
    if not isinstance(base, dict):
        return deepcopy(override) if isinstance(override, dict) else deepcopy(base)

    result = deepcopy(base)
    if not isinstance(override, dict):
        return result

    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_defaults() -> Dict[str, Any]:
    """加载默认配置文件"""
    if not DEFAULT_CONFIG_FILE.exists():
        return {}
    try:
        with DEFAULT_CONFIG_FILE.open("rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to load defaults from {DEFAULT_CONFIG_FILE}: {e}")
        return {}


class Config:
    """配置管理器"""

    _instance = None
    _config = {}

    def __init__(self):
        self._config = {}
        self._defaults = {}
        self._defaults_loaded = False

    def _ensure_defaults(self):
        if self._defaults_loaded:
            return
        self._defaults = _load_defaults()
        self._defaults_loaded = True

    async def load(self):
        """显式加载配置"""
        try:
            from app.core.storage import get_storage, LocalStorage

            self._ensure_defaults()

            storage = get_storage()
            config_data = await storage.load_config()
            from_remote = True

            # 从本地 data/config.toml 初始化后端
            if config_data is None:
                local_storage = LocalStorage()
                from_remote = False
                try:
                    # 尝试读取本地配置
                    config_data = await local_storage.load_config()
                except Exception as e:
                    logger.info(f"Failed to auto-init config from local: {e}")
                    config_data = {}

            config_data = config_data or {}
            merged = _deep_merge(self._defaults, config_data)

            # 自动回填缺失配置到存储
            should_persist = (not from_remote) or (merged != config_data)
            if should_persist:
                async with storage.acquire_lock("config_save", timeout=10):
                    await storage.save_config(merged)
                if not from_remote:
                    logger.info(
                        f"Initialized remote storage ({storage.__class__.__name__}) with config baseline."
                    )

            self._config = merged
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，格式 "section.key"
            default: 默认值
        """
        if "." in key:
            try:
                section, attr = key.split(".", 1)
                return self._config.get(section, {}).get(attr, default)
            except (ValueError, AttributeError):
                return default

        return self._config.get(key, default)

    async def update(self, new_config: dict):
        """更新配置"""
        from app.core.storage import get_storage

        storage = get_storage()
        async with storage.acquire_lock("config_save", timeout=10):
            self._ensure_defaults()
            base = _deep_merge(self._defaults, self._config or {})
            merged = _deep_merge(base, new_config or {})
            await storage.save_config(merged)
            self._config = merged


# 全局配置实例
config = Config()


def _env_key(key: str) -> str:
    """将配置 key 映射为环境变量名：section.key -> SECTION_KEY"""
    return str(key or "").strip().upper().replace(".", "_")


def _coerce_env_value(raw: str, default: Any) -> Any:
    """按 default 类型将环境变量字符串转换为目标类型。"""
    if raw is None:
        return default

    # 保留空字符串（例如清空 proxy/cf_clearance）
    if isinstance(default, str):
        return raw

    s = raw.strip()

    if isinstance(default, bool):
        if s.lower() in ("1", "true", "yes", "y", "on"):
            return True
        if s.lower() in ("0", "false", "no", "n", "off"):
            return False
        return default

    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(s)
        except Exception:
            return default

    if isinstance(default, float):
        try:
            return float(s)
        except Exception:
            return default

    if isinstance(default, (list, dict)):
        # 优先 JSON，其次逗号分隔（仅 list）
        try:
            parsed = json.loads(s)
            if isinstance(default, list) and isinstance(parsed, list):
                return parsed
            if isinstance(default, dict) and isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        if isinstance(default, list):
            if not s:
                return []
            return [item.strip() for item in s.split(",") if item.strip()]

        return default

    # default 为 None 或未知类型：直接返回原始字符串
    return raw


def get_config(key: str, default: Any = None) -> Any:
    """获取配置"""
    env_name = _env_key(key)
    if env_name in os.environ:
        return _coerce_env_value(os.environ.get(env_name), default)
    return config.get(key, default)


__all__ = ["Config", "config", "get_config"]
