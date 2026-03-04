import unittest

from fastapi.testclient import TestClient

from services.optimization.main import app as optimization_app


class TestOptimizationServiceMisc(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(optimization_app)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
