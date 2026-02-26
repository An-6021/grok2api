import unittest

from app.api.v1.image import ImageGenerationRequest, validate_generation_request
from app.core.exceptions import ValidationException


class TestImageModelCompat(unittest.TestCase):
    def test_accepts_grok_imagine_1_model(self):
        req = ImageGenerationRequest(
            prompt="draw a cat",
            model="grok-imagine-1.0",
            n=1,
            stream=False,
        )
        validate_generation_request(req)

    def test_accepts_grok_imagine_2_model(self):
        req = ImageGenerationRequest(
            prompt="draw a mountain",
            model="grok-imagine-2.0",
            n=1,
            stream=False,
        )
        validate_generation_request(req)

    def test_rejects_non_image_model(self):
        req = ImageGenerationRequest(
            prompt="hello",
            model="grok-4",
            n=1,
            stream=False,
        )
        with self.assertRaises(ValidationException):
            validate_generation_request(req)


if __name__ == "__main__":
    unittest.main()
