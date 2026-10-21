# ML module: tabular-baseline-v1

This directory contains the model supplied by the ML teammate. The public ROC-AUC reported for this version is 0.65005. It uses the 1367 tabular features; `word_count` is reserved and does not affect current predictions. Percentages have not been separately calibrated.

`Predictor(artifacts_dir)` loads the CatBoost model once. `predict(records)` accepts a nonempty list of complete flat records with all `feature_0` through `feature_1366`, each numeric or null. Optional fields: `index` and `word_count`. Unknown fields, including `target`, are rejected. It returns `{index, target}` where `target` is a probability from 0 to 1.

`predict_vector(payload, index)` accepts a little-endian float32 vector read from PostgreSQL and returns one result. The HTTP adapter converts `target` to the public field `probability`.

The example request and response files are reference cases for smoke tests. `model_info.json` includes the SHA-256 hash checked during startup. For a new model version, replace code and artifacts together, then rebuild the PostgreSQL lookup if the feature schema changes.
