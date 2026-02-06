import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.api.v1.chat import ChatCompletionRequest, _resolve_custom_personality
from app.services.grok.chat import ChatRequestBuilder, MessageExtractor


class TestCustomPersonality(unittest.TestCase):
    def test_resolve_prefers_custom_personality(self):
        req = ChatCompletionRequest(
            model="grok-3",
            messages=[{"role": "user", "content": "hi"}],
            customPersonality="A",
            systemPrompt="B",
        )
        self.assertEqual(_resolve_custom_personality(req), "A")

    def test_resolve_from_system_prompt_alias(self):
        req = ChatCompletionRequest(
            model="grok-3",
            messages=[{"role": "user", "content": "hi"}],
            system_prompt="SYS1",
        )
        self.assertEqual(_resolve_custom_personality(req), "SYS1")

        req2 = ChatCompletionRequest(
            model="grok-3",
            messages=[{"role": "user", "content": "hi"}],
            systemPrompt="SYS2",
        )
        self.assertEqual(_resolve_custom_personality(req2), "SYS2")

    def test_resolve_from_extra_system_message_alias(self):
        req = ChatCompletionRequest(
            model="grok-3",
            messages=[{"role": "user", "content": "hi"}],
            systemMessage="SYS3",
        )
        self.assertEqual(_resolve_custom_personality(req), "SYS3")

    def test_resolve_from_object_prompt(self):
        req = ChatCompletionRequest(
            model="grok-3",
            messages=[{"role": "user", "content": "hi"}],
            systemMessage={"enabled": True, "text": "SYS4"},
        )
        self.assertEqual(_resolve_custom_personality(req), "SYS4")

    def test_stream_allows_undefined_string(self):
        req = ChatCompletionRequest(
            model="grok-3",
            messages=[{"role": "user", "content": "hi"}],
            stream="[undefined]",
        )
        self.assertIsNone(req.stream)

    def test_message_extractor_system_not_in_message(self):
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ]
        self.assertEqual(MessageExtractor.extract_personality(messages), "SYS")

        text, _attachments = MessageExtractor.extract(
            messages, exclude_roles={"system", "developer"}
        )
        self.assertNotIn("system:", text)
        self.assertIn("assistant:", text)
        self.assertIn("U", text)

    def test_message_extractor_supports_object_content(self):
        messages = [
            {"role": "system", "content": {"type": "text", "text": "SYS"}},
            {"role": "user", "content": "U"},
        ]
        self.assertEqual(MessageExtractor.extract_personality(messages), "SYS")

    def test_message_extractor_supports_input_text(self):
        messages = [
            {"role": "system", "content": [{"type": "input_text", "text": "SYS"}]},
            {"role": "user", "content": "U"},
        ]
        self.assertEqual(MessageExtractor.extract_personality(messages), "SYS")

    def test_build_payload_includes_custom_personality(self):
        payload = ChatRequestBuilder.build_payload(
            message="hi",
            model="grok-3",
            custom_personality="SYS",
        )
        self.assertEqual(payload.get("customPersonality"), "SYS")

    def test_build_payload_default_from_env(self):
        key = "GROK_CUSTOM_PERSONALITY_DEFAULT"
        old = os.environ.get(key)
        os.environ[key] = "DEFAULT"
        try:
            payload = ChatRequestBuilder.build_payload(message="hi", model="grok-3")
            self.assertEqual(payload.get("customPersonality"), "DEFAULT")
        finally:
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


if __name__ == "__main__":
    unittest.main()
