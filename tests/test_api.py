
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from api.main import app, get_workflow


class ApiTests(unittest.TestCase):

    def setUp(self):
        self.workflow = Mock()
        self.workflow.invoke.return_value = {
            "status": "evidence_ready",
            "results": [{"listing_id": 123}],
        }
        app.dependency_overrides[get_workflow] = (
            lambda: self.workflow
        )
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.client.close()

    def test_health(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_evidence(self):
        question = "Find apartments in Vienna"

        response = self.client.post(
            "/evidence",
            json={"question": question},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["status"],
            "evidence_ready",
        )
        self.workflow.invoke.assert_called_once_with(question)

    def test_empty_question(self):
        response = self.client.post(
            "/evidence",
            json={"question": "   "},
        )

        self.assertEqual(response.status_code, 422)
        self.workflow.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
