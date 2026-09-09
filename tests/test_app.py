from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


TEST_DB = Path(tempfile.gettempdir()) / f"aigc_review_test_{os.getpid()}.sqlite3"
os.environ["AIGC_REVIEW_DB"] = str(TEST_DB)

from app import app  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent


class AppIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        TEST_DB.unlink(missing_ok=True)

    def predict(self, relative_path: str) -> dict:
        image_path = ROOT / relative_path
        with image_path.open("rb") as handle:
            response = self.client.post(
                "/api/predict",
                data={"file": (handle, image_path.name), "policy_profile": "operational_low_false_ai"},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")
        return response.json["result"]

    def test_health_reports_open_set_and_review_privacy(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["platform_open_set"], "platform_open_set_ensemble_v5")
        self.assertFalse(response.json["review_queue"]["stores_original_images"])

    def test_real_control_routes_to_low_risk(self) -> None:
        result = self.predict("assets/samples/realctrl_person_01.jpg")
        self.assertEqual(result["binary_label"], "real")
        self.assertEqual(result["platform_label"], "real")
        self.assertEqual(result["risk_level"], "low")

    def test_generated_sample_returns_open_set_evidence(self) -> None:
        result = self.predict("assets/samples/plt03_pr101.png")
        self.assertEqual(result["binary_label"], "generated")
        self.assertIn(result["platform_label"], {"PLT01", "PLT02", "PLT03", "PLT05", "unknown_platform"})
        self.assertIn("platform_confidence", result)
        self.assertIn("platform_drift_score", result)
        self.assertEqual(len(result["platform_probabilities"]), 4)

    def test_review_case_lifecycle(self) -> None:
        result = self.predict("assets/samples/plt03_pr101.png")
        payload = {
            "file_name": "plt03_pr101.png",
            "file_sha256": result["file_sha256"],
            "binary_label": result["binary_label"],
            "generated_probability": result["generated_probability"],
            "platform_label": result["platform_label"],
            "platform_candidate": result["platform_candidate"],
            "platform_confidence": result["platform_confidence"],
            "platform_margin": result["platform_margin"],
            "model_agreement": result["platform_model_agreement"],
            "drift_score": result["platform_drift_score"],
            "risk_level": result["risk_level"],
            "reason_codes": result["platform_rejection_reasons"],
            "user_note": "integration test",
        }
        created = self.client.post("/api/reviews", json=payload)
        self.assertEqual(created.status_code, 201)
        case_id = created.json["case"]["case_id"]
        updated = self.client.patch(
            f"/api/reviews/{case_id}",
            json={
                "status": "resolved",
                "reviewer_decision": "insufficient_evidence",
                "reviewer_note": "test complete",
            },
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json["case"]["status"], "resolved")
        exported = self.client.get("/api/reviews/export.csv")
        self.assertEqual(exported.status_code, 200)
        self.assertIn(case_id, exported.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
