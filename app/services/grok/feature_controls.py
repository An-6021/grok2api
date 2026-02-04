"""
Grok 用户 Feature Controls 服务

用于调用 gRPC-Web 接口：
  auth_mgmt.AuthManagement/UpdateUserFeatureControls

当前主要用途：批量开启 always_show_nsfw_content（NSFW / Unhinged）。
"""

from __future__ import annotations

import struct
from typing import Dict, Optional, Tuple

from curl_cffi.requests import AsyncSession

from app.core.config import get_config

UPDATE_FEATURE_CONTROLS_API = "https://grok.com/auth_mgmt.AuthManagement/UpdateUserFeatureControls"
BROWSER = "chrome136"
TIMEOUT = 30


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint must be non-negative")
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _encode_key(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_varint_field(field_number: int, value: int) -> bytes:
    return _encode_key(field_number, 0) + _encode_varint(value)


def _encode_length_delimited(field_number: int, data: bytes) -> bytes:
    return _encode_key(field_number, 2) + _encode_varint(len(data)) + data


def build_update_user_feature_controls_payload(feature_key: str, enabled: bool = True) -> bytes:
    """
    构造 UpdateUserFeatureControls 的 protobuf payload（不含 gRPC-Web 5 字节 framing）。

    通过抓包可观测到的 payload（enabled=true, key=always_show_nsfw_content）：
      0a 02 10 01 12 1a 0a 18 'always_show_nsfw_content'
    """
    key_bytes = feature_key.encode("utf-8")

    # Field 1: message { field 2: varint enabled(1/0) }
    inner_a = _encode_varint_field(2, 1 if enabled else 0)
    part_a = _encode_length_delimited(1, inner_a)

    # Field 2: message { field 1: string feature_key }
    inner_b = _encode_length_delimited(1, key_bytes)
    part_b = _encode_length_delimited(2, inner_b)

    return part_a + part_b


def build_grpc_web_frame(payload: bytes) -> bytes:
    """gRPC-Web framing: 1-byte flag + 4-byte big-endian length + payload."""
    return b"\x00" + struct.pack(">I", len(payload)) + payload


def _parse_grpc_web_trailers(body: bytes) -> Dict[str, str]:
    """解析 gRPC-Web trailers frame，提取 grpc-status / grpc-message 等字段。"""
    trailers: Dict[str, str] = {}
    offset = 0

    while offset + 5 <= len(body):
        flags = body[offset]
        length = struct.unpack(">I", body[offset + 1 : offset + 5])[0]
        offset += 5

        if offset + length > len(body):
            break

        data = body[offset : offset + length]
        offset += length

        # trailers frame: MSB set
        if flags & 0x80:
            text = data.decode("utf-8", errors="replace")
            for line in text.split("\r\n"):
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                trailers[k.strip().lower()] = v.strip()

    return trailers


class FeatureControlsService:
    """Feature controls 调用封装（gRPC-Web）。"""

    def __init__(self, proxy: Optional[str] = None):
        self.proxy = proxy or get_config("grok.base_proxy_url", "")
        self.timeout = TIMEOUT

    def _build_proxies(self) -> Optional[dict]:
        return {"http": self.proxy, "https": self.proxy} if self.proxy else None

    def _build_headers(self, token: str) -> Dict[str, str]:
        token = token[4:] if token.startswith("sso=") else token
        cf = get_config("grok.cf_clearance", "")

        cookie_parts = [f"sso={token}", f"sso-rw={token}"]
        if cf:
            cookie_parts.append(f"cf_clearance={cf}")

        return {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cache-control": "no-cache",
            "content-type": "application/grpc-web+proto",
            "origin": "https://grok.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://grok.com/",
            "sec-ch-ua": '"Google Chrome";v="136", "Chromium";v="136", "Not(A:Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
            "x-grpc-web": "1",
            "x-user-agent": "connect-es/2.1.1",
            "cookie": "; ".join(cookie_parts),
        }

    async def update_user_feature_control(
        self,
        token: str,
        feature_key: str,
        enabled: bool = True,
    ) -> Tuple[bool, str]:
        payload = build_update_user_feature_controls_payload(feature_key, enabled=enabled)
        frame = build_grpc_web_frame(payload)
        headers = self._build_headers(token)

        try:
            async with AsyncSession() as session:
                resp = await session.post(
                    UPDATE_FEATURE_CONTROLS_API,
                    headers=headers,
                    data=frame,
                    impersonate=BROWSER,
                    timeout=self.timeout,
                    proxies=self._build_proxies(),
                )

            if resp.status_code != 200:
                return False, f"http:{resp.status_code}"

            trailers = _parse_grpc_web_trailers(resp.content or b"")
            grpc_status = trailers.get("grpc-status")
            if grpc_status and grpc_status != "0":
                grpc_message = trailers.get("grpc-message", "")
                return False, f"grpc:{grpc_status}:{grpc_message}".strip(":")

            return True, "ok"
        except Exception as e:
            return False, str(e)[:80]
