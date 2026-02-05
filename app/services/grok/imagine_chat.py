"""
Imagine 2.0 Chat Completions 适配层（OpenAI 兼容）

说明：
- 一些客户端（如 Cherry Studio）会使用 `/v1/chat/completions` 来请求图片模型
- 对于 `MODEL_MODE_IMAGINE_WS` 的模型，这里改为走 WebSocket imagine 通道
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import orjson

from app.core.config import get_config
from app.core.exceptions import AppException, ErrorType, UpstreamException, ValidationException
from app.core.logger import logger
from app.services.grok.imagine_ws import imagine_ws_service
from app.services.grok.model import ModelService
from app.services.token import get_token_manager


_ASPECT_RATIO_RE = re.compile(r"^\\s*\\d+\\s*:\\s*\\d+\\s*$")
_DEFAULT_STAGE_PROGRESS = {"preview": 33, "medium": 66, "final": 99}
_DEFAULT_STAGE_NAME = {"preview": "预览", "medium": "中等", "final": "高清"}


def _sse_chunk(
    chunk_id: str,
    model: str,
    created: int,
    content: str = "",
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> str:
    """构造 OpenAI SSE chunk（与项目内 StreamProcessor 输出结构一致）。"""
    delta: Dict[str, Any] = {}
    if role:
        delta["role"] = role
        delta["content"] = ""
    elif content:
        delta["content"] = content

    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": "",
        "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}],
    }
    return f"data: {orjson.dumps(payload).decode()}\n\n"


def _extract_prompt(messages: List[Dict[str, Any]]) -> str:
    """提取最后一条 user 文本作为 prompt（避免把上下文 role 前缀拼进提示词）。"""
    for msg in reversed(messages or []):
        if (msg or {}).get("role") != "user":
            continue

        content = (msg or {}).get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "text":
                    continue
                text = item.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            if parts:
                return "\n".join(parts)

    # 兜底：返回空字符串，交由上层报错
    return ""


def _extract_aspect_ratio(image_config: Any) -> str:
    """
    从 imageConfig 里提取 aspectRatio。
    - 支持 `aspectRatio` / `aspect_ratio`
    - 非法值回退 `grok.imagine_default_aspect_ratio`
    """
    default_ar = str(get_config("grok.imagine_default_aspect_ratio", "2:3") or "").strip() or "2:3"
    if not isinstance(image_config, dict):
        return default_ar

    ar = image_config.get("aspectRatio")
    if ar is None:
        ar = image_config.get("aspect_ratio")

    if not isinstance(ar, str) or not ar.strip():
        return default_ar

    ar = ar.strip().replace(" ", "")
    if not _ASPECT_RATIO_RE.match(ar):
        return default_ar

    # 目前已验证稳定的宽高比集合（与 size 映射一致）
    allowed = {"1:1", "2:3", "3:2"}
    if ar not in allowed:
        # 一些客户端会传入 3:4/4:3 等值，这里回退默认配置，避免上游直接报错
        return default_ar if default_ar in allowed else "2:3"
    return ar


def _image_format() -> str:
    """统一从 app.image_format 判断输出：url/base64。"""
    fmt = str(get_config("app.image_format", "url") or "").strip().lower()
    if fmt in {"base64", "b64_json"}:
        return "base64"
    return "url"


def _url_to_mime(url: str) -> str:
    url = str(url or "").lower()
    if url.endswith(".png"):
        return "image/png"
    if url.endswith(".webp"):
        return "image/webp"
    if url.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def _render_markdown_images(urls: List[str], b64_list: List[str]) -> str:
    """将图片渲染为 Markdown（根据 image_format 输出 URL 或 data URI）。"""
    fmt = _image_format()
    lines: List[str] = []

    if fmt == "base64":
        for idx, b64 in enumerate(b64_list or []):
            if not isinstance(b64, str) or not b64.strip():
                continue
            mime = _url_to_mime(urls[idx] if idx < len(urls) else "")
            data_uri = f"data:{mime};base64,{b64.strip()}"
            lines.append(f"![image-{idx}]({data_uri})")
    else:
        for idx, u in enumerate(urls or []):
            if not isinstance(u, str) or not u.strip():
                continue
            lines.append(f"![image-{idx}]({u.strip()})")

    return "\n".join(lines)


class ImagineChatService:
    """Imagine WS 的 Chat Completions 适配服务。"""

    @staticmethod
    async def completions(
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = None,
        thinking: str = None,
        image_config: Any = None,
    ):
        # 校验模型
        model_info = ModelService.get(model)
        if not model_info:
            raise ValidationException(
                message=f"The model `{model}` does not exist or you do not have access to it.",
                param="model",
                code="model_not_found",
            )
        if not model_info.is_image or model_info.model_mode != "MODEL_MODE_IMAGINE_WS":
            raise ValidationException(
                message=f"The model `{model}` is not supported for imagine chat.",
                param="model",
                code="model_not_supported",
            )

        prompt = _extract_prompt(messages)
        if not prompt:
            raise ValidationException(message="No prompt found in messages", param="messages", code="empty_prompt")

        # 获取 token
        try:
            token_mgr = await get_token_manager()
            await token_mgr.reload_if_stale()
            pool_name = ModelService.pool_for_model(model)
            token = token_mgr.get_token(pool_name)
        except Exception as e:
            logger.error(f"Failed to get token: {e}")
            raise AppException(
                message="Internal service error obtaining token",
                error_type=ErrorType.SERVER.value,
                code="internal_error",
            )

        if not token:
            raise AppException(
                message="No available tokens. Please try again later.",
                error_type=ErrorType.RATE_LIMIT.value,
                code="rate_limit_exceeded",
                status_code=429,
            )

        # 参数解析
        is_stream = stream if stream is not None else get_config("grok.stream", True)
        show_think = None
        if thinking == "enabled":
            show_think = True
        elif thinking == "disabled":
            show_think = False
        if show_think is None:
            show_think = bool(get_config("grok.thinking", False))

        aspect_ratio = _extract_aspect_ratio(image_config)

        # Chat 默认只生成 1 张（作为最小期望值）
        default_n = get_config("grok.imagine_chat_default_n", 1)
        try:
            default_n = int(default_n)
        except Exception:
            default_n = 1
        n = max(1, min(default_n, 4))

        if is_stream:
            return ImagineChatService._stream(
                model=model_info.model_id,
                token=token,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                n=n,
                show_think=show_think,
            )

        # 非流式
        enable_nsfw = bool(get_config("grok.imagine_enable_nsfw", True))
        result = await imagine_ws_service.generate(
            token=token,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            n=n,
            enable_nsfw=enable_nsfw,
        )

        if not result.get("success"):
            msg = str(result.get("error") or "Image generation failed")
            code = str(result.get("error_code") or "upstream_error")
            raise AppException(
                message=msg,
                error_type=ErrorType.SERVER.value,
                code=code,
                status_code=502,
            )

        urls = list(result.get("urls") or [])
        b64_list = list(result.get("b64_list") or [])
        content = _render_markdown_images(urls, b64_list)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_info.model_id,
            "system_fingerprint": "",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content, "refusal": None, "annotations": []},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0, "image_tokens": 0},
                "completion_tokens_details": {"text_tokens": 0, "audio_tokens": 0, "reasoning_tokens": 0},
            },
        }

    @staticmethod
    async def _stream(
        model: str,
        token: str,
        prompt: str,
        aspect_ratio: str,
        n: int,
        show_think: bool,
    ) -> AsyncGenerator[str, None]:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        # 先发 role
        yield _sse_chunk(chunk_id, model, created, role="assistant")
        # 先输出一行可见内容，避免某些客户端只显示“思考中”但不渲染 <think> 内文本
        yield _sse_chunk(chunk_id, model, created, content="正在生成图片...\n")

        think_opened = False
        if show_think:
            yield _sse_chunk(chunk_id, model, created, content="<think>\n")
            think_opened = True
            yield _sse_chunk(
                chunk_id,
                model,
                created,
                content=f"正在生成图片（aspect_ratio={aspect_ratio}，n={n}）...\n",
            )

        # 阶段变化去重
        image_stage: Dict[str, str] = {}
        # 将上游不断出现的 image_id 映射为连续序号（1,2,3...）
        id_to_slot: Dict[str, int] = {}
        last_heartbeat_elapsed = -999

        enable_nsfw = bool(get_config("grok.imagine_enable_nsfw", True))
        async for item in imagine_ws_service.generate_stream(
            token=token,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            n=n,
            enable_nsfw=enable_nsfw,
        ):
            if item.get("type") == "heartbeat":
                if not show_think:
                    continue
                try:
                    elapsed = int(item.get("elapsed") or 0)
                except Exception:
                    elapsed = 0
                # 避免过于频繁刷屏：每 10 秒输出一次心跳提示
                if elapsed - last_heartbeat_elapsed >= 10:
                    last_heartbeat_elapsed = elapsed
                    yield _sse_chunk(chunk_id, model, created, content=f"仍在生成中... ({elapsed}s)\n")
                continue

            if item.get("type") == "progress":
                if not show_think:
                    continue

                image_id = str(item.get("image_id") or "")
                stage = str(item.get("stage") or "")
                if not image_id:
                    continue

                # 映射到槽位：有多少就展示多少，不做上限丢弃
                slot = id_to_slot.get(image_id)
                if slot is None:
                    slot = len(id_to_slot)
                    id_to_slot[image_id] = slot

                if image_stage.get(image_id) == stage:
                    continue
                image_stage[image_id] = stage

                progress = _DEFAULT_STAGE_PROGRESS.get(stage, 0)
                stage_name = _DEFAULT_STAGE_NAME.get(stage, stage)
                total = max(1, len(id_to_slot))
                text = f"图片 {slot + 1}/{total} - {stage_name} ({progress}%)\n"
                yield _sse_chunk(chunk_id, model, created, content=text)
                continue

            if item.get("type") == "result":
                if not item.get("success"):
                    msg = str(item.get("error") or "Image generation failed")
                    code = str(item.get("error_code") or "image_generation_failed")
                    if think_opened:
                        yield _sse_chunk(chunk_id, model, created, content="</think>\n")
                    yield _sse_chunk(chunk_id, model, created, content=f"[图片生成失败] {msg} ({code})\n")
                    yield _sse_chunk(chunk_id, model, created, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    return

                urls = list(item.get("urls") or [])
                b64_list = list(item.get("b64_list") or [])

                if think_opened:
                    yield _sse_chunk(chunk_id, model, created, content="</think>\n")

                content = _render_markdown_images(urls, b64_list)
                if content:
                    yield _sse_chunk(chunk_id, model, created, content=content + "\n")

                yield _sse_chunk(chunk_id, model, created, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return

        # 理论不应到这里，兜底
        if think_opened:
            yield _sse_chunk(chunk_id, model, created, content="</think>\n")
        yield _sse_chunk(chunk_id, model, created, content="[图片生成失败] 未收到上游结果\n")
        yield _sse_chunk(chunk_id, model, created, finish_reason="stop")
        yield "data: [DONE]\n\n"


__all__ = ["ImagineChatService"]
