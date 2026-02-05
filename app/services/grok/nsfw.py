"""
NSFW (Unhinged) 模式服务

使用 gRPC-Web 协议开启账号的 NSFW 功能。
"""

from dataclasses import dataclass
from typing import Optional

from curl_cffi.requests import AsyncSession

from app.core.config import get_config
from app.core.logger import logger
from app.services.grok.grpc_web import (
    encode_grpc_web_payload,
    parse_grpc_web_response,
    get_grpc_status,
)


NSFW_API = "https://grok.com/auth_mgmt.AuthManagement/UpdateUserFeatureControls"
BROWSER = "chrome136"
TIMEOUT = 30


@dataclass
class NSFWResult:
    """NSFW 操作结果"""

    success: bool
    http_status: int
    grpc_status: Optional[int] = None
    grpc_message: Optional[str] = None
    error: Optional[str] = None


class NSFWService:
    """NSFW 模式服务"""

    def __init__(self, proxy: str = None):
        self.proxy = proxy or get_config("grok.base_proxy_url", "")

    def _build_headers(self, token: str) -> dict:
        """构造 gRPC-Web 请求头"""
        token = token[4:] if token.startswith("sso=") else token
        cf = get_config("grok.cf_clearance", "")
        cookie = f"sso={token}; sso-rw={token}"
        if cf:
            cookie += f"; cf_clearance={cf}"

        return {
            "accept": "*/*",
            "content-type": "application/grpc-web+proto",
            "origin": "https://grok.com",
            "referer": "https://grok.com/",
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "x-grpc-web": "1",
            "x-user-agent": "connect-es/2.1.1",
            "cookie": cookie,
        }

    def _build_payload(self) -> bytes:
        """构造请求 payload（feature key 可配置）"""
        # protobuf (match captured HAR):
        # 0a 02 10 01                   -> field 1 (len=2) with inner bool=true
        # 12 xx                         -> field 2, length N
        #   0a xx <name>                -> nested message with name string
        feature_key = str(
            get_config("grok.nsfw_feature_key", "always_show_nsfw_content")
        ).strip() or "always_show_nsfw_content"
        name = feature_key.encode("utf-8")
        inner = b"\x0a" + bytes([len(name)]) + name
        protobuf = b"\x0a\x02\x10\x01\x12" + bytes([len(inner)]) + inner
        return encode_grpc_web_payload(protobuf)

    async def enable(self, token: str) -> NSFWResult:
        """为单个 token 开启 NSFW 模式"""
        headers = self._build_headers(token)
        payload = self._build_payload()
        logger.debug(
            "NSFW payload: len={} hex={}",
            len(payload),
            payload.hex(),
        )
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None

        try:
            async with AsyncSession(impersonate=BROWSER) as session:
                response = await session.post(
                    NSFW_API,
                    data=payload,
                    headers=headers,
                    timeout=TIMEOUT,
                    proxies=proxies,
                )

                if response.status_code != 200:
                    # 线上批量失败时最常见是 Cloudflare/WAF 拦截，给出可读提示便于排查。
                    snippet = ""
                    try:
                        snippet = response.content[:200].decode(
                            "utf-8", errors="replace"
                        )
                        snippet = " ".join(snippet.split())
                    except Exception:
                        snippet = ""
                    logger.warning(
                        "NSFW HTTP error: status={} ct={} cf-ray={} snippet={}",
                        response.status_code,
                        response.headers.get("content-type", ""),
                        response.headers.get("cf-ray", ""),
                        snippet,
                    )
                    return NSFWResult(
                        success=False,
                        http_status=response.status_code,
                        error=f"HTTP {response.status_code}"
                        + (f": {snippet[:120]}" if snippet else ""),
                    )

                # 解析 gRPC-Web 响应
                content_type = response.headers.get("content-type")
                _, trailers = parse_grpc_web_response(
                    response.content, content_type=content_type
                )

                grpc_status = get_grpc_status(trailers)
                logger.debug(
                    "NSFW response: http={} grpc={} msg={} trailers={}",
                    response.status_code,
                    grpc_status.code,
                    grpc_status.message,
                    trailers,
                )

                # HTTP 200 且无 grpc-status（空响应）或 grpc-status=0 都算成功
                success = grpc_status.code == -1 or grpc_status.ok

                return NSFWResult(
                    success=success,
                    http_status=response.status_code,
                    grpc_status=grpc_status.code,
                    grpc_message=grpc_status.message or None,
                )

        except Exception as e:
            logger.error(f"NSFW enable failed: {e}")
            return NSFWResult(success=False, http_status=0, error=str(e)[:100])


__all__ = ["NSFWService", "NSFWResult"]
