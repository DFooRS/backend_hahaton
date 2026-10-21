"""Apply numbered SQL migrations. Safe to rerun as a Kubernetes Job."""
import os
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]


def main():
    dsn = os.environ["DATABASE_URL"]
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(987654321)")
            cur.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())")
            for path in sorted((ROOT / "migrations").glob("[0-9]*.sql")):
                cur.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (path.name,))
                if cur.fetchone():
                    continue
                cur.execute(path.read_text(encoding="utf-8"))
                cur.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,))
                print(f"Applied {path.name}", flush=True)
    print("Migrations complete", flush=True)


if __name__ == "__main__":
    main()
