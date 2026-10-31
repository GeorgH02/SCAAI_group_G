
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

    def test_workflow_failure(self):
        self.workflow.invoke.side_effect = RuntimeError("Database unavailable")

        response = self.client.post(
            "/evidence",
            json={"question": "Find apartments in Vienna"},
        )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json(),
            {"detail": "Evidence workflow failed"},
        )


    def test_internal_evidence_is_not_exposed(self):
        self.workflow.invoke.return_value = {
            "status": "evidence_ready",
            "results": [{
                "tool": "search_listings",
                "rows": [{
                    "listing_id": 123,
                    "price": "80.00",
                    "load_id": "internal-id",
                    "name": "Private listing name",
                }],
                "evidence": [{
                    "sql": "SELECT * FROM listings",
                    "parameters": [],
                }],
                "total_matches": 1,
            }],
        }

        response = self.client.post(
            "/evidence",
            json={"question": "Find apartments in Vienna"},
        )

        self.assertEqual(response.status_code, 200)

        result = response.json()["results"][0]

        self.assertEqual(
            result["rows"],
            [{"listing_id": 123, "price": "80.00"}],
        )
        self.assertNotIn("evidence", result)
        self.assertNotIn("load_id", result["rows"][0])
        self.assertNotIn("name", result["rows"][0])


class DependencyTests(unittest.TestCase):

    def test_workflow_closes_connection(self):
        from unittest.mock import patch
        from api.dependencies import get_workflow

        connection = Mock()
        connection.execute.return_value = [("Wien Westbahnhof",)]

        with (
            patch("try_agent_db.connect", return_value=connection),
            patch(
                "api.dependencies.get_resources",
                return_value=(Mock(), Mock(), Mock()),
            ),
            patch("langchain_agent.workflow.EvidenceWorkflow") as workflow_class,
        ):
            dependency = get_workflow()
            workflow = next(dependency)

            self.assertIs(workflow, workflow_class.return_value)
            connection.close.assert_not_called()

            with self.assertRaises(StopIteration):
                next(dependency)

            connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
