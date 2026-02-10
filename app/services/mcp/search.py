"""
Grok 搜索服务

通过 GrokChatService 实现 web 搜索，保留原始引用标签。
"""

import orjson

from app.core.config import get_config
from app.core.logger import logger
from app.services.grok.services.chat import GrokChatService
from app.services.grok.models.model import ModelService
from app.services.grok.processors.base import _with_idle_timeout, _normalize_stream_line
from app.services.token import get_token_manager, EffortType


SEARCH_SYSTEM_PROMPT = """You are a web research assistant. You MUST use live web search to answer the user's query.
Always cite your sources. Structure your response with:
1. A comprehensive answer based on search results
2. A sources list: Sources:\n- [Title](URL)\n- [Title](URL)"""


class SearchResponseCollector:
    """轻量级响应收集器，不过滤 xai:tool_usage_card 等标签（保留引用信息）"""

    @staticmethod
    async def collect(stream) -> str:
        """从流式响应中收集 modelResponse.message 文本"""
        parts = []
        idle_timeout = get_config("timeout.stream_idle_timeout", 45.0)

        async for line in _with_idle_timeout(stream, idle_timeout, model="mcp-search"):
            text = _normalize_stream_line(line)
            if not text:
                continue
            try:
                data = orjson.loads(text)
            except (orjson.JSONDecodeError, ValueError):
                continue

            # 路径: data["result"]["response"]["modelResponse"]["message"]
            result = data.get("result", {})
            resp = result.get("response", {})
            mr = resp.get("modelResponse")
            if not mr:
                continue

            message = mr.get("message", "")
            if message:
                parts.append(message)

        return "".join(parts)


class GrokSearchService:
    """使用 Grok 原生 web 搜索能力执行搜索"""

    async def search(self, query: str, model: str = None) -> str:
        """
        执行搜索查询

        Args:
            query: 搜索查询
            model: 使用的模型，默认从配置读取

        Returns:
            搜索结果文本（包含引用）
        """
        model = model or get_config("mcp.model", "grok-3")

        # 解析模型 -> grok_model + model_mode
        model_info = ModelService.get(model)
        if model_info:
            grok_model = model_info.grok_model
            model_mode = model_info.model_mode
        else:
            grok_model = model
            model_mode = None

        # 获取 token（与 ChatService.completions 保持一致的路由逻辑）
        token_mgr = await get_token_manager()
        await token_mgr.reload_if_stale()

        token = None
        for pool_name in ModelService.pool_candidates_for_model(model):
            token = token_mgr.get_token(pool_name)
            if token:
                break

        if not token:
            return "Error: No available tokens. Please try again later."

        try:
            service = GrokChatService()
            stream = await service.chat(
                token=token,
                message=query,
                model=grok_model,
                mode=model_mode,
                custom_personality=SEARCH_SYSTEM_PROMPT,
            )

            result = await SearchResponseCollector.collect(stream)

            # 消耗 token 配额
            try:
                await token_mgr.consume(token, EffortType.LOW)
            except Exception as e:
                logger.warning(f"MCP search: failed to record usage: {e}")

            if not result:
                return "No results found for the query."

            logger.info(f"MCP search completed: query={query[:50]}...")
            return result

        except Exception as e:
            logger.error(f"MCP search error: {e}")
            return f"Search error: {str(e)}"


__all__ = ["GrokSearchService", "SearchResponseCollector"]
