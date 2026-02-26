import unittest
from unittest.mock import AsyncMock, patch

from app.services.grok.batch_services.nsfw import NSFWService
from app.services.reverse.utils.grpc import GrpcStatus


class _Mgr:
    def __init__(self):
        self.add_tag = AsyncMock(return_value=True)
        self.record_fail = AsyncMock(return_value=None)


class TestNsfwBatchDelay(unittest.IsolatedAsyncioTestCase):
    async def test_apply_delay_after_success(self):
        mgr = _Mgr()
        tokens = ["token-a"]

        def _cfg(key: str, default=None):
            if key == "nsfw.batch_size":
                return 10
            if key == "nsfw.apply_delay_ms":
                return 500
            if key == "proxy.browser":
                return ""
            if key == "nsfw.concurrent":
                return 5
            return default

        with patch("app.services.grok.batch_services.nsfw.get_config", side_effect=_cfg), patch(
            "app.services.grok.batch_services.nsfw.AcceptTosReverse.request",
            new=AsyncMock(return_value=None),
        ), patch(
            "app.services.grok.batch_services.nsfw.SetBirthReverse.request",
            new=AsyncMock(return_value=None),
        ), patch(
            "app.services.grok.batch_services.nsfw.NsfwMgmtReverse.request",
            new=AsyncMock(return_value=GrpcStatus(code=0, message="")),
        ), patch(
            "app.services.grok.batch_services.nsfw.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ) as sleep_mock:
            results = await NSFWService.batch(tokens, mgr)

        self.assertTrue(results["token-a"]["ok"])
        mgr.add_tag.assert_awaited_once_with("token-a", "nsfw")
        sleep_mock.assert_awaited_once_with(0.5)


if __name__ == "__main__":
    unittest.main()
