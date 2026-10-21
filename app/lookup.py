"""PostgreSQL lookup of the model's float32 feature vectors."""
import hashlib
import json

import psycopg


def features_hash(features: list[str]) -> str:
    payload = json.dumps(features, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PostgresLookup:
    def __init__(self, dsn: str, features: list[str]):
        self.dsn = dsn
        self.features = features
        self.expected_hash = features_hash(features)

    def ready(self) -> bool:
        try:
            with psycopg.connect(self.dsn, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT features_sha256, row_count FROM lookup_metadata WHERE id = 1")
                    row = cur.fetchone()
            return row is not None and row[0] == self.expected_hash and row[1] > 0
        except psycopg.Error:
            return False

    def get(self, index: str | int) -> bytes | None:
        with psycopg.connect(self.dsn, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT features_sha256 FROM lookup_metadata WHERE id = 1")
                schema = cur.fetchone()
                if schema is None or schema[0] != self.expected_hash:
                    raise RuntimeError("Lookup feature schema does not match the model")
                cur.execute("SELECT payload FROM lookup_records WHERE id = %s", (str(index),))
                row = cur.fetchone()
        return bytes(row[0]) if row else None
