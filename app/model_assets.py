"""Load model artifacts from a public HTTPS ZIP or the repository."""
import hashlib
import io
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ARTIFACTS = ROOT / "ml_backend" / "artifacts"
MAX_BUNDLE_BYTES = 100 * 1024 * 1024


def verify(artifacts: Path) -> None:
    info = json.loads((artifacts / "model_info.json").read_text(encoding="utf-8"))
    actual = hashlib.sha256((artifacts / "model.cbm").read_bytes()).hexdigest()
    if actual != info["model_sha256"]:
        raise ValueError("Model checksum does not match model_info.json")


def load_artifacts():
    url = os.getenv("MODEL_BUNDLE_URL")
    if not url:
        verify(LOCAL_ARTIFACTS)
        return LOCAL_ARTIFACTS, None
    if not url.startswith("https://"):
        raise ValueError("MODEL_BUNDLE_URL must use HTTPS")
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(MAX_BUNDLE_BYTES + 1)
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("Model bundle exceeds 100 MiB")
    temporary = tempfile.TemporaryDirectory(prefix="bankruptcy-model-")
    artifacts = Path(temporary.name)
    try:
        with ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            for name in ("model.cbm", "features.json", "model_info.json"):
                matches = [entry for entry in names if entry == name or entry.endswith("/artifacts/" + name)]
                if len(matches) != 1:
                    raise ValueError(f"Model bundle must contain exactly one {name}")
                (artifacts / name).write_bytes(archive.read(matches[0]))
        verify(artifacts)
        return artifacts, temporary
    except Exception:
        temporary.cleanup()
        raise
