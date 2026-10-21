"""Optional indexed store for TRAIN and TEST. Run once, outside HTTP requests."""
import argparse
import getpass
import json
import sqlite3
from pathlib import Path
import pyzipper

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path)
    source.add_argument("--data-zip", type=Path)
    parser.add_argument("--output", type=Path, default=Path("lookup.sqlite"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    features = json.loads((root / "artifacts" / "features.json").read_text(encoding="utf-8"))
    if args.output.exists():
        raise FileExistsError(args.output)
    temporary = args.output.with_suffix(".building.sqlite")
    if temporary.exists():
        raise FileExistsError(f"Remove incomplete build first: {temporary}")
    dtype = {f: "float32" for f in features}
    dtype["index"] = "string"
    total = 0
    with sqlite3.connect(temporary) as connection:
        connection.execute("CREATE TABLE objects (id TEXT PRIMARY KEY, payload BLOB NOT NULL)")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("features", json.dumps(features)))
        archive = pyzipper.AESZipFile(args.data_zip) if args.data_zip else None
        if archive and any(item.flag_bits & 1 for item in archive.infolist() if item.filename in {"TRAIN.csv", "TEST.csv"}):
            archive.setpassword(getpass.getpass("Data ZIP password: ").encode())
        try:
            for name in ("TRAIN.csv", "TEST.csv"):
                source_file = archive.open(name) if archive else args.data_dir / name
                try:
                    for chunk in pd.read_csv(source_file, usecols=["index"] + features, dtype=dtype, chunksize=5000):
                        if chunk["index"].isna().any():
                            raise ValueError("Missing index")
                        matrix = chunk[features].to_numpy(dtype="<f4", copy=True)
                        matrix[np.isinf(matrix)] = np.nan
                        connection.executemany("INSERT INTO objects VALUES (?, ?)", ((str(key), row.tobytes()) for key, row in zip(chunk["index"], matrix)))
                        connection.commit()
                        total += len(chunk)
                        print(f"{name}: {total} objects", flush=True)
                finally:
                    if archive:
                        source_file.close()
        finally:
            if archive:
                archive.close()
        if total != 354246:
            raise ValueError(f"Unexpected object count: {total}")
    temporary.replace(args.output)
    print(f"Ready: {args.output.resolve()}; {total} objects. target was not stored.")


if __name__ == "__main__":
    main()
