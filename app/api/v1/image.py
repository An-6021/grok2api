"""
Image Generation API 路由
"""

import asyncio
import math
import random
from typing import List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import orjson

from app.services.grok.chat import GrokChatService
from app.services.grok.model import ModelService
from app.services.grok.processor import ImageStreamProcessor, ImageCollectProcessor
from app.services.grok.imagine_ws import imagine_ws_service
from app.services.token import get_token_manager
from app.core.config import get_config
from app.core.exceptions import ValidationException, AppException, ErrorType
from app.core.logger import logger


router = APIRouter(tags=["Images"])


class ImageGenerationRequest(BaseModel):
    """图片生成请求 - OpenAI 兼容"""
    prompt: str = Field(..., description="图片描述")
    model: Optional[str] = Field("grok-imagine-1.0", description="模型名称")
    n: Optional[int] = Field(1, ge=1, le=10, description="生成数量 (1-10)")
    size: Optional[str] = Field("1024x1024", description="图片尺寸（imagine 2.0 会映射为宽高比）")
    quality: Optional[str] = Field("standard", description="图片质量 (暂不支持)")
    response_format: Optional[str] = Field("b64_json", description="响应格式")
    style: Optional[str] = Field(None, description="风格 (暂不支持)")
    stream: Optional[bool] = Field(False, description="是否流式输出")


IMAGINE_WS_MODELS = {"grok-imagine-2.0", "grok-2-image"}


def is_imagine_ws_model(model_id: str) -> bool:
    return (model_id or "").strip() in IMAGINE_WS_MODELS


def size_to_aspect_ratio(size: str) -> str:
    """
    将 OpenAI 的 size 映射为 imagine 的 aspect_ratio。
    未命中时回退 `grok.imagine_default_aspect_ratio`（默认 2:3）。
    """
    size = str(size or "").strip()
    size_map = {
        "1024x1024": "1:1",
        "1024x1536": "2:3",
        "1536x1024": "3:2",
        "512x512": "1:1",
        "256x256": "1:1",
    }
    if size in size_map:
        return size_map[size]
    cfg = str(get_config("grok.imagine_default_aspect_ratio", "2:3") or "").strip()
    return cfg or "2:3"


def sse_event(event: str, data: dict) -> str:
    """构建 SSE 响应"""
    return f"event: {event}\ndata: {orjson.dumps(data).decode()}\n\n"


def validate_request(request: ImageGenerationRequest):
    """验证请求参数"""
    # 验证模型 - 通过 is_image 检查
    model_info = ModelService.get(request.model)
    if not model_info or not model_info.is_image:
        # 获取支持的图片模型列表
        image_models = [m.model_id for m in ModelService.MODELS if m.is_image]
        raise ValidationException(
            message=f"The model `{request.model}` is not supported for image generation. Supported: {image_models}",
            param="model",
            code="model_not_supported"
        )
    
    # 验证 prompt
    if not request.prompt or not request.prompt.strip():
        raise ValidationException(
            message="Prompt cannot be empty",
            param="prompt",
            code="empty_prompt"
        )
    
    # 验证 n 参数范围
    if request.n < 1 or request.n > 10:
        raise ValidationException(
            message="n must be between 1 and 10",
            param="n",
            code="invalid_n"
        )
    
    # 流式只支持 n=1 或 n=2
    if request.stream and request.n not in [1, 2]:
        # Imagine WS 模型允许任意 n（最终返回按实际上游生成的数量）
        model_info = ModelService.get(request.model)
        if not (model_info and is_imagine_ws_model(model_info.model_id)):
            raise ValidationException(
                message="Streaming is only supported when n=1 or n=2",
                param="stream",
                code="invalid_stream_n"
            )

    # response_format（仅当用户显式传入时校验）
    if hasattr(request, "model_fields_set") and "response_format" in request.model_fields_set:
        if request.response_format and request.response_format not in {"b64_json", "url"}:
            raise ValidationException(
                message="response_format must be 'b64_json' or 'url'",
                param="response_format",
                code="invalid_response_format"
            )


def resolve_response_format(request: ImageGenerationRequest) -> str:
    """
    解析输出格式：
    - 优先使用请求中的 response_format（若用户显式传入）
    - 否则使用配置 app.image_format（url/base64）
    """
    if hasattr(request, "model_fields_set") and "response_format" in request.model_fields_set:
        fmt = (request.response_format or "").strip().lower()
        if fmt in {"b64_json", "url"}:
            return fmt

    cfg = str(get_config("app.image_format", "url") or "").strip().lower()
    if cfg in {"base64", "b64_json"}:
        return "b64_json"
    return "url"


async def call_grok(token: str, prompt: str, model_info, output_format: str) -> List[str]:
    """
    调用 Grok 获取图片，返回 base64 或 url 列表
    """
    chat_service = GrokChatService()
    
    response = await chat_service.chat(
        token=token,
        message=f"Image Generation:{prompt}",
        model=model_info.grok_model,
        mode=model_info.model_mode,
        think=False,
        stream=True
    )
    
    # 收集图片
    processor = ImageCollectProcessor(model_info.model_id, token, output_format=output_format)
    return await processor.process(response)


async def call_imagine_ws(token: str, prompt: str, aspect_ratio: str, n: int, output_format: str) -> List[str]:
    """调用 Imagine WebSocket 获取图片，返回 base64 或 url 列表"""
    enable_nsfw = bool(get_config("grok.imagine_enable_nsfw", True))
    result = await imagine_ws_service.generate(
        token=token,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        n=n,
        enable_nsfw=enable_nsfw,
    )
    if not result.get("success"):
        raise AppException(
            message=f"Imagine generation failed: {result.get('error') or 'unknown error'}",
            error_type=ErrorType.SERVER.value,
            code=str(result.get("error_code") or "upstream_error"),
            status_code=502,
        )

    if output_format == "url":
        return list(result.get("urls") or [])
    return list(result.get("b64_list") or [])


@router.post("/images/generations")
async def create_image(request: ImageGenerationRequest):
    """
    Image Generation API
    
    流式响应格式:
    - event: image_generation.partial_image
    - event: image_generation.completed
    
    非流式响应格式:
    - {"created": ..., "data": [{"b64_json": "..."}], "usage": {...}}
    """
    # stream 默认为 false
    if request.stream is None:
        request.stream = False
    
    # 参数验证
    validate_request(request)

    output_format = resolve_response_format(request)
    
    # 获取 token
    try:
        token_mgr = await get_token_manager()
        await token_mgr.reload_if_stale()
        pool_name = ModelService.pool_for_model(request.model)
        token = token_mgr.get_token(pool_name)
    except Exception as e:
        logger.error(f"Failed to get token: {e}")
        raise AppException(
            message="Internal service error obtaining token",
            error_type=ErrorType.SERVER.value,
            code="internal_error"
        )

    if not token:
        raise AppException(
            message="No available tokens. Please try again later.",
            error_type=ErrorType.RATE_LIMIT.value,
            code="rate_limit_exceeded",
            status_code=429
        )
    
    # 获取模型信息
    model_info = ModelService.get(request.model)

    # ===================== Imagine 2.0 (WebSocket) =====================
    if model_info and is_imagine_ws_model(model_info.model_id):
        # 仅当用户显式传入 size 时才做映射，否则使用配置默认比例（避免被 Pydantic 默认值误导）
        if hasattr(request, "model_fields_set") and "size" in request.model_fields_set:
            aspect_ratio = size_to_aspect_ratio(request.size)
        else:
            aspect_ratio = str(get_config("grok.imagine_default_aspect_ratio", "2:3") or "").strip() or "2:3"

        # 流式模式（复用本项目 SSE 事件定义）
        if request.stream:
            stage_progress = {"preview": 33, "medium": 66, "final": 99}

            async def _gen():
                id_to_index: dict[str, int] = {}

                async for item in imagine_ws_service.generate_stream(
                    token=token,
                    prompt=request.prompt,
                    aspect_ratio=aspect_ratio,
                    n=request.n,
                ):
                    if item.get("type") == "heartbeat":
                        # SSE 心跳（注释行），防止长时间无数据导致客户端断开/卡住
                        elapsed = item.get("elapsed")
                        if elapsed is None:
                            yield ": ping\n\n"
                        else:
                            yield f": ping {elapsed}\n\n"
                        continue

                    if item.get("type") == "progress":
                        image_id = str(item.get("image_id") or "")
                        stage = str(item.get("stage") or "")
                        if not image_id:
                            continue

                        out_index = id_to_index.get(image_id)
                        if out_index is None:
                            out_index = len(id_to_index)
                            id_to_index[image_id] = out_index

                        yield sse_event(
                            "image_generation.partial_image",
                            {
                                "type": "image_generation.partial_image",
                                "b64_json": "",
                                "url": "",
                                "index": out_index,
                                "progress": stage_progress.get(stage, 0),
                            },
                        )
                        continue

                    if item.get("type") == "result":
                        if not item.get("success"):
                            msg = str(item.get("error") or "Image generation failed")
                            code = str(item.get("error_code") or "image_generation_failed")
                            yield sse_event(
                                "image_generation.error",
                                {"type": "image_generation.error", "message": msg, "code": code},
                            )
                            return

                        values = item.get("urls") if output_format == "url" else item.get("b64_list")
                        values = values or []
                        valid = [v for v in values if isinstance(v, str) and v.strip()]
                        if not valid:
                            yield sse_event(
                                "image_generation.error",
                                {
                                    "type": "image_generation.error",
                                    "message": "图片生成失败：未获取到有效图片资源",
                                    "code": "image_generation_failed",
                                },
                            )
                            return

                        # 最终返回多少就给多少（不再按 request.n 截断）
                        for index, val in enumerate(valid):
                            payload = {
                                "type": "image_generation.completed",
                                "b64_json": "" if output_format == "url" else val,
                                "url": val if output_format == "url" else "",
                                "index": index,
                                "usage": {
                                    "total_tokens": 50,
                                    "input_tokens": 25,
                                    "output_tokens": 25,
                                    "input_tokens_details": {"text_tokens": 5, "image_tokens": 20},
                                },
                            }
                            yield sse_event("image_generation.completed", payload)
                        return

            return StreamingResponse(
                _gen(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # 非流式模式：单次 WS，最终返回“实际上游生成的全部图片”
        n = request.n
        errors: List[str] = []

        try:
            all_images = await call_imagine_ws(token, request.prompt, aspect_ratio, n, output_format)
        except Exception as e:
            logger.error(f"Imagine WS call failed: {e}")
            errors.append(str(e)[:200])
            all_images = []

        valid_images = [
            img for img in all_images
            if isinstance(img, str) and img.strip() and img.strip().lower() != "error"
        ]

        if not valid_images:
            reason = errors[0] if errors else "no valid images returned"
            raise AppException(
                message=f"Image generation failed: {reason}",
                error_type=ErrorType.SERVER.value,
                code="image_generation_failed",
                status_code=502,
            )

        # 最终返回多少就给多少（不再按 request.n 截断/抽样）
        selected_images = list(valid_images)

        import time
        if output_format == "url":
            data = [{"url": img} for img in selected_images]
        else:
            data = [{"b64_json": img} for img in selected_images]

        return JSONResponse(
            content={
                "created": int(time.time()),
                "data": data,
                "usage": {
                    "total_tokens": 0 * len(selected_images),
                    "input_tokens": 0,
                    "output_tokens": 0 * len(selected_images),
                    "input_tokens_details": {"text_tokens": 0, "image_tokens": 0},
                },
            }
        )
    
    # 流式模式
    if request.stream:
        chat_service = GrokChatService()
        response = await chat_service.chat(
            token=token,
            message=f"Image Generation:{request.prompt}",
            model=model_info.grok_model,
            mode=model_info.model_mode,
            think=False,
            stream=True
        )
        
        processor = ImageStreamProcessor(model_info.model_id, token, n=request.n, output_format=output_format)
        return StreamingResponse(
            processor.process(response),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    # 非流式模式
    n = request.n
    
    calls_needed = (n + 1) // 2 
    errors: List[str] = []
    
    if calls_needed == 1:
        # 单次调用
        try:
            all_images = await call_grok(token, request.prompt, model_info, output_format)
        except Exception as e:
            logger.error(f"Grok image call failed: {e}")
            errors.append(str(e)[:200])
            all_images = []
    else:
        # 并发调用
        tasks = [
            call_grok(token, request.prompt, model_info, output_format)
            for _ in range(calls_needed)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 收集成功的图片
        all_images = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Concurrent call failed: {result}")
                errors.append(str(result)[:200])
            elif isinstance(result, list):
                all_images.extend(result)
    
    # 过滤空图片 / 非法占位（避免出现 /v1/files/image/ 的裂图请求）
    valid_images = [
        img for img in all_images
        if isinstance(img, str) and img.strip() and img.strip().lower() != "error"
    ]

    if not valid_images:
        reason = errors[0] if errors else "no valid images returned"
        raise AppException(
            message=f"Image generation failed: {reason}",
            error_type=ErrorType.SERVER.value,
            code="image_generation_failed",
            status_code=502,
        )

    # 随机选取最多 n 张图片（不再填充 error，避免客户端裂图）
    selected_images = random.sample(valid_images, min(n, len(valid_images)))
    
    # 构建响应
    import time
    if output_format == "url":
        data = [{"url": img} for img in selected_images]
    else:
        data = [{"b64_json": img} for img in selected_images]
    
    return JSONResponse(content={
        "created": int(time.time()),
        "data": data,
        "usage": {
            "total_tokens": 0 * len(selected_images),
            "input_tokens": 0,
            "output_tokens": 0 * len(selected_images),
            "input_tokens_details": {"text_tokens": 0, "image_tokens": 0}
        }
    })


__all__ = ["router"]
