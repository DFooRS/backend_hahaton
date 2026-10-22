import csv
import io
import json
import os
import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.model_assets import artifacts_dir

SMOKE = artifacts_dir() / "smoke_cases.json"


@unittest.skipUnless(SMOKE.is_file(), "set ML_ARTIFACTS_DIR to the ML artifacts folder")
class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(SMOKE.read_text(encoding="utf-8"))
        cls.ctx = TestClient(app)
        cls.client = cls.ctx.__enter__()
        cls.features = cls.client.app.state.predictor.features

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def csv_bytes(self, records):
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
        return out.getvalue().encode()

    def post_csv(self, data):
        return self.client.post("/api/v1/predict/csv", files={"file": ("x.csv", data, "text/csv")})

    def test_by_id_returns_stored_score(self):
        for case in self.cases:
            r = self.client.post("/api/v1/predict/by-id", json={"index": case["record"]["index"]})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["probability"], case["expected"])

    def test_by_id_unknown_is_404(self):
        self.assertEqual(self.client.post("/api/v1/predict/by-id", json={"index": "no-such-id"}).status_code, 404)

    def test_csv_matches_reference(self):
        records = [c["record"] for c in self.cases]
        r = self.post_csv(self.csv_bytes(records))
        self.assertEqual(r.status_code, 200)
        for got, case in zip(r.json()["predictions"], self.cases):
            self.assertAlmostEqual(got["probability"], case["expected"], delta=2e-7 + 1e-6 * case["expected"])

    def test_csv_without_word_count_and_index(self):
        records = [{f: c["record"][f] for f in self.features} for c in self.cases[:2]]
        self.assertEqual(self.post_csv(self.csv_bytes(records)).status_code, 200)

    def test_csv_more_than_1000_rows_is_batched(self):
        base = self.cases[0]["record"]
        records = [dict(base, index=str(i)) for i in range(1001)]
        r = self.post_csv(self.csv_bytes(records))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["predictions"]), 1001)

    def test_csv_missing_column_and_empty_value_rejected(self):
        record = dict(self.cases[0]["record"])
        broken = {k: v for k, v in record.items() if k != "feature_5"}
        self.assertEqual(self.post_csv(self.csv_bytes([broken])).status_code, 422)
        record["feature_7"] = ""
        self.assertEqual(self.post_csv(self.csv_bytes([record])).status_code, 422)

    def test_records_invalid_rejected(self):
        self.assertEqual(self.client.post("/api/v1/predict/records", json={"records": [{"target": 1}]}).status_code, 422)

    def test_health_and_metrics(self):
        for path in ("/actuator/health/liveness", "/actuator/health/readiness"):
            self.assertEqual(self.client.get(path).json(), {"status": "UP"})
        self.assertIn("backend_http_requests_total", self.client.get("/actuator/prometheus").text)


if __name__ == "__main__":
    unittest.main()
