import io
import os
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from app.model_assets import LOCAL_ARTIFACTS, load_artifacts


class ModelAssetsTest(unittest.TestCase):
    def test_public_bundle_extracts_and_verifies_model(self):
        bundle = io.BytesIO()
        with ZipFile(bundle, "w") as archive:
            for name in ("model.cbm", "features.json", "model_info.json"):
                archive.write(LOCAL_ARTIFACTS / name, f"ml_backend/artifacts/{name}")
        bundle.seek(0)
        with patch.dict(os.environ, {"MODEL_BUNDLE_URL": "https://example.test/model.zip"}), patch(
            "urllib.request.urlopen", return_value=io.BytesIO(bundle.getvalue())
        ):
            artifacts, temporary = load_artifacts()
            try:
                self.assertTrue((artifacts / "model.cbm").is_file())
            finally:
                temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
