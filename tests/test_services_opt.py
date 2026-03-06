"""
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

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
