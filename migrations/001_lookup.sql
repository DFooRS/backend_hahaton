CREATE TABLE IF NOT EXISTS lookup_records (
    id text PRIMARY KEY,
    payload bytea NOT NULL
);

CREATE TABLE IF NOT EXISTS lookup_metadata (
    id smallint PRIMARY KEY CHECK (id = 1),
    features_sha256 char(64) NOT NULL,
    row_count integer NOT NULL CHECK (row_count > 0),
    imported_at timestamptz NOT NULL DEFAULT now()
);
