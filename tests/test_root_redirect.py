import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from main import create_app


class TestRootRedirect(unittest.TestCase):
    def test_root_redirects_to_admin(self):
        app = create_app()
        client = TestClient(app)
        resp = client.get("/", follow_redirects=False)
        self.assertIn(resp.status_code, (301, 302, 307, 308))
        self.assertEqual(resp.headers.get("location"), "/admin")


if __name__ == "__main__":
    unittest.main()
