import unittest
from unittest.mock import AsyncMock, patch

from app.core.exceptions import UpstreamException
from app.services.reverse.app_chat import AppChatReverse
from app.services.reverse.media_post import MediaPostReverse


class _FakeResponse:
    def __init__(self, status_code: int, text_body: str = "", lines=None):
        self.status_code = status_code
        self._text = text_body
        self._lines = list(lines or [])
        self.headers = {}
        self.content = text_body.encode("utf-8")

    async def text(self):
        return self._text

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.post_calls = 0

    async def post(self, *args, **kwargs):
        self.post_calls += 1
        if not self._responses:
            raise RuntimeError("no prepared response")
        return self._responses.pop(0)

    async def close(self):
        return None


class TestAgeVerifyRetry(unittest.IsolatedAsyncioTestCase):
    async def _no_retry(self, func, *args, **kwargs):
        return await func(*args)

    async def test_app_chat_retries_once_after_age_verify(self):
        session = _FakeSession(
            [
                _FakeResponse(403, "forbidden"),
                _FakeResponse(200, lines=["ok"]),
            ]
        )
        with patch(
            "app.services.reverse.app_chat.SetBirthReverse.request",
            new=AsyncMock(return_value=None),
        ) as birth_mock, patch(
            "app.services.reverse.app_chat.retry_on_status",
            new=self._no_retry,
        ):
            stream = await AppChatReverse.request(
                session=session,
                token="token",
                message="hello",
                model="grok-3",
            )
            lines = [line async for line in stream]

        self.assertEqual(lines, ["ok"])
        self.assertEqual(session.post_calls, 2)
        birth_mock.assert_awaited_once()

    async def test_app_chat_raises_after_second_403(self):
        session = _FakeSession(
            [
                _FakeResponse(403, "forbidden"),
                _FakeResponse(403, "forbidden again"),
            ]
        )
        with patch(
            "app.services.reverse.app_chat.SetBirthReverse.request",
            new=AsyncMock(return_value=None),
        ) as birth_mock, patch(
            "app.services.reverse.app_chat.retry_on_status",
            new=self._no_retry,
        ):
            with self.assertRaises(UpstreamException) as ctx:
                await AppChatReverse.request(
                    session=session,
                    token="token",
                    message="hello",
                    model="grok-3",
                )

        self.assertTrue(ctx.exception.details.get("age_verify_attempted"))
        self.assertEqual(session.post_calls, 2)
        birth_mock.assert_awaited_once()

    async def test_media_post_retries_once_after_age_verify(self):
        session = _FakeSession(
            [
                _FakeResponse(403, "forbidden"),
                _FakeResponse(200, "ok"),
            ]
        )
        with patch(
            "app.services.reverse.media_post.SetBirthReverse.request",
            new=AsyncMock(return_value=None),
        ) as birth_mock, patch(
            "app.services.reverse.media_post.retry_on_status",
            new=self._no_retry,
        ):
            resp = await MediaPostReverse.request(
                session=session,
                token="token",
                mediaType="MEDIA_POST_TYPE_VIDEO",
                mediaUrl="",
                prompt="hello",
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(session.post_calls, 2)
        birth_mock.assert_awaited_once()

    async def test_media_post_raises_after_second_403(self):
        session = _FakeSession(
            [
                _FakeResponse(403, "forbidden"),
                _FakeResponse(403, "forbidden again"),
            ]
        )
        with patch(
            "app.services.reverse.media_post.SetBirthReverse.request",
            new=AsyncMock(return_value=None),
        ) as birth_mock, patch(
            "app.services.reverse.media_post.retry_on_status",
            new=self._no_retry,
        ):
            with self.assertRaises(UpstreamException) as ctx:
                await MediaPostReverse.request(
                    session=session,
                    token="token",
                    mediaType="MEDIA_POST_TYPE_VIDEO",
                    mediaUrl="",
                    prompt="hello",
                )

        self.assertTrue(ctx.exception.details.get("age_verify_attempted"))
        self.assertEqual(session.post_calls, 2)
        birth_mock.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
