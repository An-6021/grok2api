"""
Grok Imagine 2.0 WebSocket 图片生成服务

参考仓库：Bailing258/imagine2api

实现要点：
- 通过 WebSocket `wss://grok.com/ws/imagine/listen` 直连 Grok Imagine
- 支持自动年龄验证（依赖 `grok.cf_clearance`）
- 支持代理（HTTP/HTTPS/SOCKS4/SOCKS5，依赖 aiohttp-socks）
- 生成结果同时返回 URL（保存到本地缓存目录）与 base64（b64_json）
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import re
import ssl
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import orjson

try:
    import aiohttp
except Exception as e:  # pragma: no cover
    aiohttp = None  # type: ignore
    _AIOHTTP_IMPORT_ERROR = e

try:
    from aiohttp_socks import ProxyConnector
except Exception:  # pragma: no cover
    ProxyConnector = None  # type: ignore

from curl_cffi.requests import AsyncSession

from app.core.config import get_config
from app.core.logger import logger
from app.services.grok.assets import DownloadService
from app.services.token import get_token_manager


SET_BIRTH_DATE_API = "https://grok.com/rest/auth/set-birth-date"
DEFAULT_WS_URL = "wss://grok.com/ws/imagine/listen"
BROWSER = "chrome136"

# Token tags
AGE_VERIFIED_TAG = "age_verified"

# 并发限制（避免大量 WS 连接导致资源飙升）
DEFAULT_MAX_CONCURRENT = 20
_IMAGINE_SEMAPHORE = asyncio.Semaphore(DEFAULT_MAX_CONCURRENT)
_IMAGINE_SEM_VALUE = DEFAULT_MAX_CONCURRENT


def _get_imagine_semaphore() -> asyncio.Semaphore:
    global _IMAGINE_SEMAPHORE, _IMAGINE_SEM_VALUE
    value = get_config("performance.imagine_max_concurrent", DEFAULT_MAX_CONCURRENT)
    try:
        value = int(value)
    except Exception:
        value = DEFAULT_MAX_CONCURRENT
    value = max(1, value)
    if value != _IMAGINE_SEM_VALUE:
        _IMAGINE_SEM_VALUE = value
        _IMAGINE_SEMAPHORE = asyncio.Semaphore(value)
    return _IMAGINE_SEMAPHORE


@dataclass
class ImageProgress:
    """单张图片进度（按 image_id 聚合）"""

    image_id: str
    stage: str = "preview"  # preview -> medium -> final
    blob: str = ""
    blob_size: int = 0
    url: str = ""
    is_final: bool = False


@dataclass
class GenerationProgress:
    """整体生成进度"""

    total: int = 4
    images: Dict[str, ImageProgress] = field(default_factory=dict)
    completed: int = 0

    def check_blocked(self) -> bool:
        """检测 blocked：出现 medium 但一直没有 final"""
        has_medium = any(img.stage == "medium" for img in self.images.values())
        has_final = any(img.is_final for img in self.images.values())
        return has_medium and not has_final


StreamCallback = Callable[[ImageProgress, GenerationProgress], Awaitable[None]]


class GrokImagineWSService:
    """Grok Imagine 2.0 WebSocket 调用封装"""

    def __init__(self):
        self._ssl_context = ssl.create_default_context()
        # 兼容不同格式：/images/<id>.jpg|jpeg|png|webp 或无扩展名
        self._url_pattern = re.compile(
            r"/image[s]?/([A-Za-z0-9_-]{8,})(?:\.(?:png|jpe?g|webp))?",
            flags=re.IGNORECASE,
        )

    def _get_ws_url(self) -> str:
        url = str(get_config("grok.imagine_ws_url", DEFAULT_WS_URL) or "").strip()
        return url or DEFAULT_WS_URL

    def _get_timeout(self) -> int:
        timeout = get_config("grok.imagine_timeout", None)
        if timeout is None:
            timeout = get_config("grok.timeout", 120)
        try:
            timeout = int(timeout)
        except Exception:
            timeout = 120
        return max(15, timeout)

    def _get_proxy_url(self) -> str:
        return str(get_config("grok.base_proxy_url", "") or "").strip()

    def _get_ws_headers(self, token: str) -> Dict[str, str]:
        token = token[4:] if token.startswith("sso=") else token
        cf = str(get_config("grok.cf_clearance", "") or "").strip()

        cookie_parts = [f"sso={token}", f"sso-rw={token}"]
        if cf:
            cookie_parts.append(f"cf_clearance={cf}")

        return {
            "Cookie": "; ".join(cookie_parts),
            "Origin": "https://grok.com",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def _extract_image_id(self, url: str) -> Optional[str]:
        raw = str(url or "").strip()
        if not raw:
            return None

        # 优先从 path 提取，避免 querystring 影响
        try:
            path = urlparse(raw).path or raw
        except Exception:
            path = raw

        match = self._url_pattern.search(path)
        if match:
            return match.group(1)

        # 兜底：取最后一段文件名（去扩展名）
        seg = path.rsplit("/", 1)[-1].strip()
        if not seg:
            return None
        if "." in seg:
            seg = seg.split(".", 1)[0]
        seg = re.sub(r"[^A-Za-z0-9_-]", "", seg)
        return seg or None

    def _is_final_image(self, url: str, blob_size: int) -> bool:
        # 经验规则：最终高清通常为 jpeg/webp 且体积较大（blob_size 为 base64 字符数）
        u = str(url or "").lower()
        if blob_size > 200000:
            return True
        if u.endswith((".jpg", ".jpeg", ".webp")) and blob_size > 100000:
            return True
        if u.endswith(".png") and blob_size > 140000:
            return True
        return False

    @staticmethod
    def _guess_ext(url: str, is_final: bool) -> str:
        try:
            path = urlparse(str(url or "")).path
        except Exception:
            path = str(url or "")

        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if ext in {"jpg", "jpeg", "png", "webp", "gif"}:
            return ext
        return "jpg" if is_final else "png"

    def _get_connector(self):
        """根据配置构建连接器（支持 SOCKS/HTTP 代理）。"""
        if aiohttp is None:  # pragma: no cover
            raise RuntimeError(f"aiohttp 未安装: {_AIOHTTP_IMPORT_ERROR}")

        proxy_url = self._get_proxy_url()
        if not proxy_url:
            return aiohttp.TCPConnector(ssl=self._ssl_context)

        if ProxyConnector is None:
            # aiohttp-socks 未安装时，仍允许 HTTP 代理（socks 将不支持）
            logger.warning("aiohttp-socks 未安装，SOCKS 代理将不可用")
            return aiohttp.TCPConnector(ssl=self._ssl_context)

        # aiohttp-socks 支持 http/https/socks4/socks5
        return ProxyConnector.from_url(proxy_url, ssl=self._ssl_context)

    async def _ensure_age_verified(self, token: str) -> None:
        """确保完成年龄验证（仅在需要时调用）。"""
        if not get_config("grok.imagine_auto_age_verify", True):
            return

        token = token[4:] if token.startswith("sso=") else token
        mgr = await get_token_manager()
        info = mgr.find(token)
        if info and AGE_VERIFIED_TAG in (info.tags or []):
            return

        cf = str(get_config("grok.cf_clearance", "") or "").strip()
        if not cf:
            logger.warning("未配置 grok.cf_clearance，无法自动年龄验证（将继续尝试生成）")
            return

        birth_date = str(get_config("grok.imagine_birth_date", "2001-01-01T16:00:00.000Z") or "").strip()
        if not birth_date:
            birth_date = "2001-01-01T16:00:00.000Z"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/133.0.0.0 Safari/537.36"
            ),
            "Origin": "https://grok.com",
            "Referer": "https://grok.com/",
            "Accept": "*/*",
            "Cookie": f"sso={token}; sso-rw={token}; cf_clearance={cf}",
            "Content-Type": "application/json",
        }

        proxy = self._get_proxy_url()
        proxies = {"http": proxy, "https": proxy} if proxy else None

        try:
            async with AsyncSession() as session:
                resp = await session.post(
                    SET_BIRTH_DATE_API,
                    headers=headers,
                    json={"birthDate": birth_date},
                    impersonate=BROWSER,
                    timeout=30,
                    proxies=proxies,
                )

            if resp.status_code == 200:
                logger.info("年龄验证成功")
                await mgr.add_tag(token, AGE_VERIFIED_TAG)
            else:
                try:
                    text = resp.text[:200]
                except Exception:
                    text = ""
                logger.warning(f"年龄验证失败: {resp.status_code} {text}")
        except Exception as e:
            logger.warning(f"年龄验证请求异常: {str(e)[:120]}")

    async def _save_best_images(
        self,
        progress: GenerationProgress,
        n: Optional[int] = None,
        prefer_image_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        将图片保存到本地缓存目录，同时返回 URL 与 base64 列表。

        说明：
        - 流式场景下，prefer_image_ids 用于保证最终输出顺序与进度 index 对齐（优先保存这些 id）。
        - 非流式场景下，prefer_image_ids 可以为空，按“final 优先 + blob 最大”挑选。
        """
        dl_service = DownloadService()
        image_dir = dl_service.image_dir
        image_dir.mkdir(parents=True, exist_ok=True)

        saved_ids: set[str] = set()
        urls: List[str] = []
        b64_list: List[str] = []

        limit = None
        if n is not None:
            try:
                limit = int(n)
            except Exception:
                limit = None
        if limit is not None:
            limit = max(1, limit)

        def _save_one(img: ImageProgress) -> None:
            if img.image_id in saved_ids:
                return
            if limit is not None and len(saved_ids) >= limit:
                return
            if not img.blob:
                return
            try:
                ext = self._guess_ext(img.url, img.is_final)
                safe_id = img.image_id.replace("/", "").strip()
                filename = f"imagine2-{safe_id}.{ext}"
                file_path = image_dir / filename

                if not file_path.exists():
                    blob = img.blob.strip()
                    if "base64," in blob:
                        blob = blob.split("base64,", 1)[1].strip()
                    data = base64.b64decode(blob)
                    file_path.write_bytes(data)

                urls.append(dl_service.get_public_url(f"/image/{filename}"))
                b64_list.append(img.blob)
                saved_ids.add(img.image_id)
            except Exception as e:
                logger.warning(f"保存 imagine 图片失败: {str(e)[:120]}")

        # 1) 优先保存 prefer_image_ids（用于流式 index 对齐）
        if prefer_image_ids:
            for image_id in prefer_image_ids:
                if limit is not None and len(saved_ids) >= limit:
                    break
                img = progress.images.get(image_id)
                if img:
                    _save_one(img)

        # 2) 补齐剩余名额：final 优先，如果没有则使用最大的版本
        candidates = sorted(progress.images.values(), key=lambda x: (x.is_final, x.blob_size), reverse=True)
        for img in candidates:
            if limit is not None and len(saved_ids) >= limit:
                break
            _save_one(img)

        # 触发缓存上限检查（异步）
        try:
            asyncio.create_task(dl_service.check_limit())
        except Exception:
            pass

        return urls, b64_list

    async def generate(
        self,
        token: str,
        prompt: str,
        aspect_ratio: str = "2:3",
        n: int = 4,
        enable_nsfw: Optional[bool] = None,
        stream_callback: Optional[StreamCallback] = None,
        prefer_image_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        生成图片（单次 WS 调用）

        返回：
        - success=true: {urls, b64_list, count}
        - success=false: {error_code, error}
        """
        if aiohttp is None:  # pragma: no cover
            raise RuntimeError(f"aiohttp 未安装: {_AIOHTTP_IMPORT_ERROR}")

        # n 在这里作为“最小期望数量”（进度展示/blocked 判定参考），不作为上限；
        # 最终返回会把一次 WS 过程中实际生成到的图片全部返回。
        n = max(1, int(n or 1))
        enable_nsfw = bool(get_config("grok.imagine_enable_nsfw", True) if enable_nsfw is None else enable_nsfw)

        await self._ensure_age_verified(token)

        ws_url = self._get_ws_url()
        headers = self._get_ws_headers(token)
        timeout_sec = self._get_timeout()

        request_id = str(uuid.uuid4())
        msg = {
            "type": "conversation.item.create",
            "timestamp": int(time.time() * 1000),
            "item": {
                "type": "message",
                "content": [
                    {
                        "requestId": request_id,
                        "text": prompt,
                        "type": "input_text",
                        "properties": {
                            "section_count": 0,
                            "is_kids_mode": False,
                            "enable_nsfw": enable_nsfw,
                            "skip_upsampler": False,
                            "is_initial": False,
                            "aspect_ratio": aspect_ratio,
                        },
                    }
                ],
            },
        }

        receive_timeout = float(get_config("grok.imagine_receive_timeout_sec", 5) or 5)
        blocked_after = float(get_config("grok.imagine_blocked_after_sec", 15) or 15)
        idle_finish = float(get_config("grok.imagine_idle_finish_sec", 10) or 10)

        progress = GenerationProgress(total=n)
        last_activity = time.time()
        start_time = time.time()
        medium_received_time: Optional[float] = None
        error_info: Optional[Dict[str, Any]] = None
        ws_close_code: Optional[int] = None
        ws_close_reason: str = ""
        ws_transport_error: str = ""
        seen_msg_types: set[str] = set()
        parse_failures = 0
        async with _get_imagine_semaphore():
            connector = self._get_connector()

            try:
                async with aiohttp.ClientSession(connector=connector) as session:
                    proxy_url = self._get_proxy_url()
                    ws_kwargs = {
                        "headers": headers,
                        "heartbeat": 20,
                        "receive_timeout": timeout_sec,
                    }

                    # aiohttp 原生 HTTP 代理支持（socks 走 aiohttp-socks connector）
                    if proxy_url and ProxyConnector is None and proxy_url.startswith(("http://", "https://")):
                        ws_kwargs["proxy"] = proxy_url

                    async with session.ws_connect(ws_url, **ws_kwargs) as ws:
                        await ws.send_json(msg)
                        logger.info(f"Imagine WS 已发送请求: n={n} aspect_ratio={aspect_ratio}")

                        while time.time() - start_time < timeout_sec:
                            try:
                                ws_msg = await asyncio.wait_for(ws.receive(), timeout=receive_timeout)
                            except asyncio.TimeoutError:
                                # 超时分支：判断 blocked 或已完成
                                if medium_received_time and progress.completed == 0:
                                    if (time.time() - medium_received_time) > max(1.0, blocked_after - 5):
                                        return {
                                            "success": False,
                                            "error_code": "blocked",
                                            "error": "生成被阻止，无法获取最终图片",
                                        }

                                if progress.completed > 0 and (time.time() - last_activity) > idle_finish:
                                    break
                                continue

                            if ws_msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                                last_activity = time.time()
                                try:
                                    raw = ws_msg.data
                                    if isinstance(raw, str):
                                        raw = raw.encode("utf-8")
                                    elif isinstance(raw, bytearray):
                                        raw = bytes(raw)
                                    if not isinstance(raw, (bytes, bytearray)):
                                        continue
                                    data = orjson.loads(raw)
                                except Exception:
                                    parse_failures += 1
                                    continue

                                if not isinstance(data, dict):
                                    continue

                                msg_type = str(data.get("type") or "").strip()
                                if msg_type:
                                    seen_msg_types.add(msg_type)

                                # 兼容：image 可能在嵌套结构中（conversation.item.* 等）
                                img_data: Optional[Dict[str, Any]] = None
                                if msg_type == "image":
                                    img_data = data
                                else:
                                    for key in ("item", "data", "payload", "message"):
                                        sub = data.get(key)
                                        if isinstance(sub, dict) and sub.get("type") == "image":
                                            img_data = sub
                                            break
                                    if img_data is None:
                                        content = data.get("content")
                                        if isinstance(content, list):
                                            for it in content:
                                                if isinstance(it, dict) and it.get("type") == "image":
                                                    img_data = it
                                                    break
                                        if img_data is None and isinstance(data.get("item"), dict):
                                            item = data.get("item") or {}
                                            content = item.get("content")
                                            if isinstance(content, list):
                                                for it in content:
                                                    if isinstance(it, dict) and it.get("type") == "image":
                                                        img_data = it
                                                        break

                                if img_data is not None:
                                    blob = (
                                        img_data.get("blob")
                                        or img_data.get("b64_json")
                                        or img_data.get("b64")
                                        or ""
                                    )
                                    url = (
                                        img_data.get("url")
                                        or img_data.get("image_url")
                                        or img_data.get("imageUrl")
                                        or img_data.get("blob_url")
                                        or img_data.get("blobUrl")
                                        or ""
                                    )

                                    if isinstance(url, dict):
                                        url = url.get("url") or ""

                                    if not isinstance(blob, str):
                                        blob = ""
                                    if not isinstance(url, str):
                                        url = ""

                                    if not blob and not url:
                                        continue

                                    image_id = str(
                                        img_data.get("image_id")
                                        or img_data.get("imageId")
                                        or img_data.get("id")
                                        or ""
                                    ).strip()
                                    if not image_id and url:
                                        image_id = self._extract_image_id(url) or ""

                                    # 最后兜底：用 url/blob 计算一个稳定 id
                                    if not image_id:
                                        if url:
                                            image_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
                                        elif blob:
                                            image_id = hashlib.sha1(blob[:1024].encode("utf-8")).hexdigest()[:16]
                                        else:
                                            image_id = uuid.uuid4().hex

                                    blob_size = len(blob) if blob else 0

                                    stage = str(img_data.get("stage") or "").lower().strip()
                                    if stage not in {"preview", "medium", "final"}:
                                        is_final = self._is_final_image(url, blob_size)
                                        if is_final:
                                            stage = "final"
                                        elif blob_size > 30000:
                                            stage = "medium"
                                        else:
                                            stage = "preview"
                                    else:
                                        is_final = stage == "final"

                                    if stage == "medium" and medium_received_time is None:
                                        medium_received_time = time.time()

                                    img = ImageProgress(
                                        image_id=image_id,
                                        stage=stage,
                                        blob=blob,
                                        blob_size=blob_size,
                                        url=url,
                                        is_final=is_final,
                                    )

                                    existing = progress.images.get(image_id)
                                    # 只在“更好”的结果到来时更新：final 优先，其次更大的 blob
                                    if (
                                        (existing is None)
                                        or (img.is_final and not existing.is_final)
                                        or (img.blob_size > (existing.blob_size if existing else 0))
                                    ):
                                        progress.images[image_id] = img
                                        progress.completed = len([x for x in progress.images.values() if x.is_final])

                                        if stream_callback:
                                            try:
                                                await stream_callback(img, progress)
                                            except Exception as e:
                                                logger.warning(f"Imagine 流式回调异常: {str(e)[:120]}")

                                # 兼容不同错误结构
                                if "error" in msg_type:
                                    err_code = str(data.get("err_code") or data.get("code") or "")
                                    err_msg = str(data.get("err_msg") or data.get("message") or "")

                                    err_obj = data.get("error")
                                    if isinstance(err_obj, dict):
                                        err_code = err_code or str(err_obj.get("code") or "")
                                        err_msg = err_msg or str(err_obj.get("message") or "")
                                    elif isinstance(err_obj, str):
                                        err_msg = err_msg or err_obj

                                    if err_code or err_msg:
                                        error_info = {
                                            "error_code": err_code or "upstream_error",
                                            "error": err_msg or "upstream error",
                                        }

                                        # token 失效：记录 401 失败次数（达到阈值后自动标记 expired）
                                        if (err_code or "").strip() == "unauthorized":
                                            try:
                                                mgr = await get_token_manager()
                                                await mgr.record_fail(token, 401, err_msg or "unauthorized")
                                            except Exception:
                                                pass

                                        if (err_code or "").strip() in {"rate_limit_exceeded", "unauthorized"}:
                                            break

                                # blocked 检测：有 medium 但超过阈值仍无 final
                                if medium_received_time and progress.completed == 0:
                                    if (time.time() - medium_received_time) > blocked_after:
                                        return {
                                            "success": False,
                                            "error_code": "blocked",
                                            "error": "生成被阻止，无法获取最终图片",
                                        }

                            elif ws_msg.type == aiohttp.WSMsgType.CLOSE:
                                try:
                                    ws_close_code = int(ws_msg.data) if ws_msg.data is not None else ws.close_code
                                except Exception:
                                    ws_close_code = ws.close_code
                                ws_close_reason = str(getattr(ws_msg, "extra", "") or "")
                                break
                            elif ws_msg.type == aiohttp.WSMsgType.ERROR:
                                try:
                                    ws_transport_error = str(ws.exception() or "")
                                except Exception:
                                    ws_transport_error = ""
                                ws_close_code = ws.close_code
                                break
                            elif ws_msg.type in (aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED):
                                ws_close_code = ws.close_code
                                break

            except aiohttp.ClientError as e:
                return {"success": False, "error": f"连接失败: {str(e)}"}
            except Exception as e:
                logger.error(f"Imagine WS 生成异常: {e}")
                return {"success": False, "error": f"生成异常: {str(e)[:200]}"}

        # 生成结果落盘 + 返回
        if progress.images:
            # 最终返回多少就给多少：不限制数量
            urls, b64_list = await self._save_best_images(progress, None, prefer_image_ids=prefer_image_ids)
            if urls:
                return {"success": True, "urls": urls, "b64_list": b64_list, "count": len(urls)}

        if error_info:
            return {"success": False, **error_info}

        if progress.check_blocked():
            return {"success": False, "error_code": "blocked", "error": "生成被阻止，无法获取最终图片"}

        if ws_close_code or ws_transport_error:
            msg = f"WebSocket 已关闭: code={ws_close_code or ''} {ws_close_reason}".strip()
            if ws_transport_error:
                msg = f"{msg} {ws_transport_error}".strip()
            return {"success": False, "error_code": "ws_closed", "error": msg}

        if seen_msg_types or parse_failures:
            info = ",".join(sorted(list(seen_msg_types))[:10])
            details = []
            if info:
                details.append(f"types={info}")
            if parse_failures:
                details.append(f"parse_failures={parse_failures}")
            suffix = f" ({' '.join(details)})" if details else ""
            return {"success": False, "error_code": "no_image_data", "error": f"未收到图片数据{suffix}"}

        return {"success": False, "error_code": "no_image_data", "error": "未收到图片数据"}

    async def generate_stream(
        self,
        token: str,
        prompt: str,
        aspect_ratio: str = "2:3",
        n: int = 2,
        enable_nsfw: Optional[bool] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式生成图片（返回结构化事件，路由层负责 SSE 格式化）。

        Yields:
        - {"type":"progress", "image_id":..., "stage":..., "is_final":..., "completed":..., "total":...}
        - {"type":"result", "success":true/false, ...}
        """
        queue: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()
        prefer_ids: List[str] = []

        async def _cb(img: ImageProgress, prog: GenerationProgress):
            # 记录“被选中”的 image_id，用于最终输出顺序对齐
            if img.image_id and img.image_id not in prefer_ids and len(prefer_ids) < (n or 1):
                prefer_ids.append(img.image_id)
            await queue.put(
                {
                    "type": "progress",
                    "image_id": img.image_id,
                    "stage": img.stage,
                    "is_final": img.is_final,
                    "completed": prog.completed,
                    "total": prog.total,
                }
            )

        async def _runner():
            try:
                # 关键：generate 内部有超时，但 ws_connect 握手阶段可能卡住，
                # 这里再套一层 wait_for 兜底，避免流式一直无输出导致客户端“卡住”。
                base_timeout = float(self._get_timeout())
                buffer_sec = float(get_config("grok.imagine_stream_timeout_buffer_sec", 15) or 15)
                runner_timeout = max(30.0, base_timeout + max(0.0, buffer_sec))

                try:
                    result = await asyncio.wait_for(
                        self.generate(
                            token=token,
                            prompt=prompt,
                            aspect_ratio=aspect_ratio,
                            n=n,
                            enable_nsfw=enable_nsfw,
                            stream_callback=_cb,
                            prefer_image_ids=prefer_ids,
                        ),
                        timeout=runner_timeout,
                    )
                except asyncio.TimeoutError:
                    result = {"success": False, "error_code": "timeout", "error": "生成超时"}
                await queue.put({"type": "result", **result})
            except Exception as e:
                await queue.put({"type": "result", "success": False, "error": str(e)[:200]})
            finally:
                done.set()

        runner_task = asyncio.create_task(_runner())
        start_time = time.time()
        heartbeat_sec = float(get_config("grok.imagine_stream_heartbeat_sec", 5) or 5)
        heartbeat_sec = max(0.0, heartbeat_sec)

        try:
            while True:
                if done.is_set() and queue.empty():
                    break

                if heartbeat_sec <= 0:
                    item = await queue.get()
                else:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=heartbeat_sec)
                    except asyncio.TimeoutError:
                        # 心跳：让上层有机会持续输出“仍在生成中...”，避免客户端一直停在思考态
                        yield {"type": "heartbeat", "elapsed": int(time.time() - start_time)}
                        continue

                yield item
                if item.get("type") == "result":
                    break
        finally:
            if not runner_task.done():
                runner_task.cancel()
                with contextlib.suppress(Exception):
                    await runner_task


imagine_ws_service = GrokImagineWSService()

__all__ = ["imagine_ws_service", "GrokImagineWSService"]
