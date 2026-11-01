
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from api.main import app, get_answerer, get_workflow


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



class AnswerApiTests(unittest.TestCase):
    ANSWER = {
        "status": "evidence_ready",
        "answer": {
            "answer": "Listing 7 is in Alsergrund [listing 7].",
            "verified": True,
            "pii_redacted": False,
            "citations": [{"sentence": "Listing 7 is in Alsergrund [listing 7].", "citations": ["listing 7"],
                           "evidence": ["Listing 7. It is in the district Alsergrund."]}],
            "removed": [{"sentence": "It has a pool.", "reason": "not entailed"}],
        },
        "reviews": [{"scope": "Matching listings", "review_id": 9, "listing_id": 7, "review_date": "2026-05-01",
                     "language": "en", "text": "Very quiet.", "score": 0.9}],
        "results": [{"tool": "search_listings", "rows": [{"listing_id": 7, "name": "secret", "neighbourhood": "Alsergrund"}],
                     "evidence": [{"sql": "SELECT", "parameters": []}], "notices": []}],
    }

    def answer(self, result, **body):
        calls = []
        app.dependency_overrides[get_answerer] = lambda: (lambda *args: calls.append(args) or result)
        try:
            with TestClient(app) as client:
                response = client.post("/answer", json={"question": "Quiet flats in Alsergrund", **body})
        finally:
            app.dependency_overrides.clear()
        return response, calls

    def test_answer_returns_checked_statements_reviews_and_filtered_evidence(self):
        response, calls = self.answer(self.ANSWER, new_years_eve_year=2026)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(calls, [("Quiet flats in Alsergrund", "2026")])
        self.assertEqual(body["answer"], "Listing 7 is in Alsergrund [listing 7].")
        self.assertTrue(body["verified"])
        self.assertEqual(body["citations"][0]["citations"], ["listing 7"])
        self.assertEqual(body["removed"][0]["sentence"], "It has a pool.")
        self.assertEqual(body["reviews"][0]["text"], "Very quiet.")
        self.assertNotIn("score", body["reviews"][0])
        self.assertNotIn("secret", str(body["evidence"]))
        self.assertNotIn("SELECT", str(body["evidence"]))
        self.assertEqual(body["notices"], [])

    def test_blocked_and_clarification_results_have_a_message(self):
        response, _ = self.answer({"status": "blocked", "results": [], "message": "Off topic.",
                                   "category": "off_topic", "reason": "not about Vienna accommodation",
                                   "masked_question": "Write code for <PERSON_1>"})
        body = response.json()
        self.assertEqual(body["status"], "blocked")
        self.assertEqual(body["message"], "Off topic.")
        self.assertEqual((body["guard_category"], body["guard_reason"]), ("off_topic", "not about Vienna accommodation"))
        self.assertEqual(body["masked_question"], "Write code for <PERSON_1>")
        self.assertIsNone(body["answer"])
        response, _ = self.answer({"clarification": "Which year?"})
        self.assertEqual(response.json()["status"], "clarification_needed")

    def test_invalid_year_and_pipeline_failure(self):
        response, calls = self.answer(self.ANSWER, new_years_eve_year=1999)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(calls, [])
        app.dependency_overrides[get_answerer] = lambda: (lambda *args: (_ for _ in ()).throw(RuntimeError("DSN secret")))
        try:
            with TestClient(app) as client:
                response = client.post("/answer", json={"question": "Quiet flats"})
        finally:
            app.dependency_overrides.clear()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret", response.text)

if __name__ == "__main__":
    unittest.main()
