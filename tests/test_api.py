import csv
import io
import json
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parents[1]


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads((ROOT / "ml_backend/example_request.json").read_text())
        cls.expected = json.loads((ROOT / "ml_backend/example_response.json").read_text())
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def test_json_matches_ml_reference(self):
        response = self.client.post("/api/v1/predict/records", json={"records": self.records})
        self.assertEqual(response.status_code, 200)
        for got, expected in zip(response.json()["predictions"], self.expected):
            self.assertEqual(got["index"], expected["index"])
            self.assertAlmostEqual(got["probability"], expected["target"], places=12)

    def test_csv_matches_json(self):
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=self.records[0].keys())
        writer.writeheader()
        writer.writerow(self.records[0])
        response = self.client.post(
            "/api/v1/predict/csv",
            files={"file": ("sample.csv", output.getvalue().encode(), "text/csv")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(response.json()["predictions"][0]["probability"], self.expected[0]["target"], places=12)

    def test_invalid_features_rejected(self):
        response = self.client.post("/api/v1/predict/records", json={"records": [{"target": 1}]})
        self.assertEqual(response.status_code, 422)

    def test_index_lookup(self):
        features = self.client.app.state.predictor.features
        vector = np.asarray([self.records[0][name] for name in features], dtype="<f4").tobytes()

        class FakeLookup:
            def ready(self):
                return True

            def get(self, index):
                return vector if str(index) == "194357" else None

        original = self.client.app.state.lookup
        self.client.app.state.lookup = FakeLookup()
        try:
            response = self.client.post("/api/v1/predict/by-id", json={"index": "194357"})
            self.assertEqual(response.status_code, 200)
            self.assertAlmostEqual(response.json()["probability"], self.expected[0]["target"], places=12)
            self.assertEqual(self.client.post("/api/v1/predict/by-id", json={"index": "does-not-exist"}).status_code, 404)
        finally:
            self.client.app.state.lookup = original

    def test_index_unavailable_without_database(self):
        self.assertEqual(self.client.post("/api/v1/predict/by-id", json={"index": "194357"}).status_code, 503)


if __name__ == "__main__":
    unittest.main()
