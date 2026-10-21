"""Import TRAIN and TEST feature vectors into PostgreSQL in one transaction."""
import json
import os
import ssl
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import pyzipper
import certifi

from app.lookup import features_hash
from app.model_assets import load_artifacts

CHUNK_SIZE = 1000
MAX_ARCHIVE_BYTES = 5 * 1024 * 1024 * 1024
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


@contextmanager
def csv_sources():
    archive_url = os.getenv("DATA_ZIP_URL")
    train_url = os.getenv("TRAIN_CSV_URL")
    test_url = os.getenv("TEST_CSV_URL")
    data_dir = os.getenv("DATA_DIR")
    if data_dir:
        if archive_url or train_url or test_url:
            raise ValueError("Use DATA_DIR or one remote data source")
        directory = Path(data_dir)
        with (directory / "TRAIN.csv").open("rb") as train, (directory / "TEST.csv").open("rb") as test:
            yield (("TRAIN.csv", train), ("TEST.csv", test))
    elif archive_url:
        if train_url or test_url:
            raise ValueError("Use DATA_ZIP_URL or separate CSV URLs, not both")
        if not archive_url.startswith("https://"):
            raise ValueError("DATA_ZIP_URL must use HTTPS")
        with tempfile.TemporaryDirectory(prefix="bankruptcy-data-") as directory:
            archive_path = Path(directory) / "data.zip"
            total = 0
            with urllib.request.urlopen(archive_url, timeout=60, context=TLS_CONTEXT) as response, archive_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("Data archive exceeds 5 GiB")
                    output.write(chunk)
            with pyzipper.AESZipFile(archive_path) as archive:
                password = os.getenv("DATA_ZIP_PASSWORD")
                if any(item.flag_bits & 1 for item in archive.infolist() if item.filename in {"TRAIN.csv", "TEST.csv"}):
                    if not password:
                        raise ValueError("DATA_ZIP_PASSWORD is required for the encrypted data archive")
                    archive.setpassword(password.encode())
                with archive.open("TRAIN.csv") as train, archive.open("TEST.csv") as test:
                    yield (("TRAIN.csv", train), ("TEST.csv", test))
    elif train_url and test_url:
        if not train_url.startswith("https://") or not test_url.startswith("https://"):
            raise ValueError("CSV URLs must use HTTPS")
        with urllib.request.urlopen(train_url, timeout=60, context=TLS_CONTEXT) as train, urllib.request.urlopen(test_url, timeout=60, context=TLS_CONTEXT) as test:
            yield (("TRAIN.csv", train), ("TEST.csv", test))
    else:
        raise ValueError("Set DATA_ZIP_URL or both TRAIN_CSV_URL and TEST_CSV_URL")


def main():
    artifacts, temporary = load_artifacts()
    try:
        features = json.loads((artifacts / "features.json").read_text(encoding="utf-8"))
        dtype = {name: "float32" for name in features}
        dtype["index"] = "string"
        expected_rows = int(os.getenv("EXPECTED_LOOKUP_ROWS", "354246"))
        total = 0
        with csv_sources() as sources, psycopg.connect(os.environ["DATABASE_URL"]) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(987654322)")
                cur.execute("TRUNCATE lookup_records")
                cur.execute("DELETE FROM lookup_metadata")
                with cur.copy("COPY lookup_records (id, payload) FROM STDIN") as copy:
                    for filename, source in sources:
                        for chunk in pd.read_csv(source, usecols=["index"] + features, dtype=dtype, chunksize=CHUNK_SIZE):
                            if chunk["index"].isna().any():
                                raise ValueError(f"{filename}: missing index")
                            matrix = chunk[features].to_numpy(dtype="<f4", copy=True)
                            matrix[np.isinf(matrix)] = np.nan
                            for key, row in zip(chunk["index"], matrix):
                                copy.write_row((str(key), row.tobytes()))
                            total += len(chunk)
                            if total % 10000 < CHUNK_SIZE:
                                print(f"Imported {total} rows", flush=True)
                if total != expected_rows:
                    raise ValueError(f"Unexpected row count: {total}; expected {expected_rows}")
                cur.execute(
                    "INSERT INTO lookup_metadata (id, features_sha256, row_count) VALUES (1, %s, %s)",
                    (features_hash(features), total),
                )
        print(f"Lookup ready: {total} rows", flush=True)
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
