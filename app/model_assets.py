"""Locate and sanity-check the ML artifacts directory (stored on the server, never in Git/image)."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "ml_backend" / "artifacts"
REQUIRED = (
    "features.json", "raw_lightgbm.txt", "joint_lightgbm.txt", "text_C1.joblib", "text_C4.joblib",
    "text_svd.joblib", "linear.joblib", "linear_preprocessing.joblib", "scores.sqlite", "profiles.sqlite",
)
# Files small enough to hash on every start; the full check is `python verify_package.py`.
HASHED_AT_START = REQUIRED[:-2]


def artifacts_dir() -> Path:
    return Path(os.getenv("ML_ARTIFACTS_DIR") or DEFAULT_ARTIFACTS).resolve()


def verify(artifacts: Path) -> None:
    missing = [name for name in REQUIRED if not (artifacts / name).is_file()]
    if missing:
        raise FileNotFoundError(f"ML artifacts missing in {artifacts}: {', '.join(missing)}")
    manifest_path = ROOT / "ml_backend" / "manifest.json"
    if os.getenv("VERIFY_ARTIFACTS_ON_START", "1") == "0" or not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in HASHED_AT_START:
        digest = hashlib.sha256((artifacts / name).read_bytes()).hexdigest()
        if digest != manifest.get(f"artifacts/{name}"):
            raise ValueError(f"Checksum mismatch: {name}")


def load_artifacts() -> Path:
    artifacts = artifacts_dir()
    verify(artifacts)
    return artifacts
