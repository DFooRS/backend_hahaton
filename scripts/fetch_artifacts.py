"""Download the ML artifacts tar from S3 (streamed, no temp file) and verify every file against manifest.json."""
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "features.json", "raw_lightgbm.txt", "joint_lightgbm.txt", "text_C1.joblib", "text_C4.joblib",
    "text_svd.joblib", "linear.joblib", "linear_preprocessing.joblib", "scores.sqlite", "profiles.sqlite",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(url: str, target: Path) -> None:
    manifest = json.loads((ROOT / "ml_backend" / "manifest.json").read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, tarfile.open(fileobj=response, mode="r|") as tar:
        for member in tar:
            name = member.name.removeprefix("./")
            if not member.isfile() or name not in NAMES:
                continue  # ignore anything else in the tar, incl. paths with directories
            member.name = name
            tar.extract(member, target, filter="data")
    for name in NAMES:
        path = target / name
        if not path.is_file():
            sys.exit(f"Missing in archive: {name}")
        if sha256(path) != manifest[f"artifacts/{name}"]:
            sys.exit(f"Checksum mismatch: {name}")
    print(f"OK: {len(NAMES)} artifacts verified in {target}")


if __name__ == "__main__":
    main(os.environ.get("MODEL_BUNDLE_URL") or sys.argv[1], Path(sys.argv[2] if len(sys.argv) > 2 else "/models/artifacts"))
