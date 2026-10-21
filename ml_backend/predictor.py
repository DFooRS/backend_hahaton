"""CPU inference. Load once at application startup; no training here."""
import json
import numbers
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


class Predictor:
    def __init__(self, artifacts_dir, thread_count=2):
        artifacts = Path(artifacts_dir)
        self.features = json.loads((artifacts / "features.json").read_text(encoding="utf-8"))
        self.feature_set = set(self.features)
        self.thread_count = thread_count
        self.model = CatBoostClassifier()
        self.model.load_model(str(artifacts / "model.cbm"))
        if list(self.model.feature_names_) != self.features:
            raise ValueError("Model and features.json disagree")

    def predict(self, records):
        """Flat JSON-like records -> [{index, target}]. word_count is reserved."""
        if not isinstance(records, list) or not records:
            raise ValueError("records must be a nonempty list")
        rows = []
        for number, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Record {number}: expected an object")
            missing = self.feature_set - record.keys()
            extra = record.keys() - self.feature_set - {"index", "word_count"}
            if missing or extra:
                raise ValueError(f"Record {number}: missing={sorted(missing)[:5]}, unknown={sorted(extra)[:5]}")
            if "word_count" in record and record["word_count"] is not None and not isinstance(record["word_count"], str):
                raise ValueError("word_count must be a string or null")
            if record.get("index") is not None and (isinstance(record["index"], bool) or not isinstance(record["index"], (str, int))):
                raise ValueError("index must be a string, integer or null")
            values = []
            for feature in self.features:
                value = record[feature]
                if value is not None and (isinstance(value, bool) or not isinstance(value, numbers.Real)):
                    raise ValueError(f"Record {number}: {feature} must be numeric or null")
                values.append(np.nan if value is None else value)
            rows.append(values)
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.asarray(rows, dtype=np.float32)
        matrix[np.isinf(matrix)] = np.nan
        return self._predict_matrix(matrix, [r.get("index") for r in records])

    def _predict_matrix(self, matrix, identifiers):
        frame = pd.DataFrame(matrix, columns=self.features)
        probabilities = self.model.predict_proba(frame, thread_count=self.thread_count)[:, 1]
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError("Invalid model output")
        return [{"index": key, "target": float(p)} for key, p in zip(identifiers, probabilities)]

    def predict_vector(self, payload: bytes, index: str | int):
        matrix = np.frombuffer(payload, dtype="<f4")
        if len(matrix) != len(self.features):
            raise ValueError("Invalid lookup record")
        return self._predict_matrix(matrix.reshape(1, -1), [str(index)])[0]
