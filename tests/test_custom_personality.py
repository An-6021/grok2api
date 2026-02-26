import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

try:
    from app.services.grok.services.chat import MessageExtractor
    from app.services.grok.services.chat import GrokChatService
except ModuleNotFoundError:
    MessageExtractor = None
    GrokChatService = None

try:
    from app.services.reverse.app_chat import AppChatReverse
except ModuleNotFoundError:
    AppChatReverse = None

try:
    from app.api.v1.chat import ChatCompletionRequest, _resolve_custom_personality
except ModuleNotFoundError:
    ChatCompletionRequest = None
    _resolve_custom_personality = None


class TestCustomPersonality(unittest.TestCase):
    @unittest.skipIf(ChatCompletionRequest is None, "fastapi not installed")
    def test_resolve_prefers_custom_personality(self):
        req = ChatCompletionRequest(
            model="grok-3",
            messages=[{"role": "user", "content": "hi"}],
            customPersonality="A",
            systemPrompt="B",
        )
        self.assertEqual(_resolve_custom_personality(req), "A")

    @unittest.skipIf(ChatCompletionRequest is None, "fastapi not installed")
    def test_resolve_from_alias_and_stream_undefined(self):
        req = ChatCompletionRequest(
            model="grok-3",
            messages=[{"role": "user", "content": "hi"}],
            systemPrompt="SYS",
            stream="[undefined]",
        )
        self.assertEqual(_resolve_custom_personality(req), "SYS")
        self.assertIsNone(req.stream)

    @unittest.skipIf(MessageExtractor is None, "runtime deps not installed")
    def test_extract_personality_supports_structured_content(self):
        messages = [
            {"role": "system", "content": {"type": "text", "text": "S1"}},
            {"role": "developer", "content": [{"type": "input_text", "text": "S2"}]},
            {"role": "user", "content": "U"},
        ]
        self.assertEqual(MessageExtractor.extract_personality(messages), "S1\n\nS2")

    @unittest.skipIf(AppChatReverse is None, "runtime deps not installed")
    def test_build_payload_includes_custom_personality(self):
        payload = AppChatReverse.build_payload(
            message="hello",
            model="grok-3",
            custom_personality="SYS",
        )
        self.assertEqual(payload.get("customPersonality"), "SYS")


@unittest.skipIf(GrokChatService is None, "runtime deps not installed")
class TestCustomPersonalityDefault(unittest.IsolatedAsyncioTestCase):
    async def test_default_personality_used_when_missing(self):
        service = GrokChatService()

        with patch(
            "app.services.grok.services.chat.ModelService.get",
            return_value=SimpleNamespace(
                grok_model="grok-3",
                model_mode="MODEL_MODE_GROK_3",
            ),
        ), patch(
            "app.services.grok.services.chat.MessageExtractor.extract_personality",
            return_value="",
        ), patch(
            "app.services.grok.services.chat.MessageExtractor.extract",
            return_value=("hello", [], []),
        ), patch(
            "app.services.grok.services.chat.get_config",
            side_effect=lambda key, default=None: (
                "DEFAULT_PERSONA"
                if key == "app.custom_personality_default"
                else (False if key == "app.stream" else default)
            ),
        ), patch.object(
            service,
            "chat",
            AsyncMock(return_value="OK"),
        ) as chat_mock:
            await service.chat_openai(
                token="t",
                model="grok-3",
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
            )

        self.assertEqual(chat_mock.await_args.kwargs.get("custom_personality"), "DEFAULT_PERSONA")

if __name__ == "__main__":
    unittest.main()
