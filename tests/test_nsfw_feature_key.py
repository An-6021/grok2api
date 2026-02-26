import unittest
from unittest.mock import patch

from app.services.reverse.nsfw_mgmt import NsfwMgmtReverse


class _FakeResponse:
    def __init__(self):
        self.status_code = 200
        self.headers = {
            "content-type": "application/grpc-web+proto",
            "grpc-status": "0",
        }
        self.content = b""


class _CaptureSession:
    def __init__(self):
        self.data = None

    async def post(self, *args, **kwargs):
        self.data = kwargs.get("data")
        return _FakeResponse()


class TestNsfwFeatureKey(unittest.IsolatedAsyncioTestCase):
    async def _no_retry(self, func, *args, **kwargs):
        return await func(*args)

    async def test_payload_uses_configurable_feature_key(self):
        session = _CaptureSession()

        def _cfg(key: str, default=None):
            if key == "proxy.base_proxy_url":
                return ""
            if key == "nsfw.timeout":
                return 1
            if key == "proxy.browser":
                return ""
            if key == "nsfw.feature_key":
                return "custom_feature_key"
            return default

        with patch("app.services.reverse.nsfw_mgmt.get_config", side_effect=_cfg), patch(
            "app.services.reverse.nsfw_mgmt.retry_on_status",
            new=self._no_retry,
        ):
            await NsfwMgmtReverse.request(session, "token")

        self.assertIsInstance(session.data, (bytes, bytearray))
        self.assertIn(b"custom_feature_key", bytes(session.data))


if __name__ == "__main__":
    unittest.main()
