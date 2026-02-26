import unittest

try:
    from app.services.grok.services.chat import MessageExtractor
except ModuleNotFoundError:
    MessageExtractor = None

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


if __name__ == "__main__":
    unittest.main()
