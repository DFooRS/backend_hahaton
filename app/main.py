"""HTTP adapter for the versioned ML Predictor."""
import csv
import io
import logging
import math
import os
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, field_validator

from ml_backend.predictor import Predictor
from app.lookup import PostgresLookup
from app.model_assets import load_artifacts

MAX_CSV_ROWS = 1000
MAX_CSV_BYTES = 32 * 1024 * 1024
logger = logging.getLogger(__name__)


class IdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: StrictStr | StrictInt

    @field_validator("index")
    @classmethod
    def check_index(cls, value):
        if isinstance(value, bool) or str(value).strip() == "":
            raise ValueError("index must be a nonempty string or integer")
        return value


class RecordsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict]


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts, temporary = load_artifacts()
    app.state.predictor = Predictor(
        artifacts,
        thread_count=int(os.getenv("MODEL_THREADS", "2")),
    )
    dsn = os.getenv("DATABASE_URL")
    app.state.lookup = PostgresLookup(dsn, app.state.predictor.features) if dsn else None
    try:
        yield
    finally:
        if temporary:
            temporary.cleanup()


app = FastAPI(title="Bankruptcy prediction API", version="1.0.0", lifespan=lifespan)
origins = [s.strip() for s in os.getenv("CORS_ORIGINS", "https://web.team-frontend.hack.week-infra.ru").split(",") if s.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


def public_result(row):
    return {"index": row["index"], "probability": row["target"]}


def predict_records(request: Request, records: list[dict]):
    if not 1 <= len(records) <= MAX_CSV_ROWS:
        raise HTTPException(422, f"Provide 1 to {MAX_CSV_ROWS} records")
    try:
        return [public_result(row) for row in request.app.state.predictor.predict(records)]
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed") from exc


@app.get("/health")
def health(request: Request):
    lookup = request.app.state.lookup
    return {"model_ready": True, "lookup_ready": bool(lookup and lookup.ready())}


@app.post("/api/v1/predict/by-id")
def predict_by_id(payload: IdRequest, request: Request):
    lookup = request.app.state.lookup
    if lookup is None or not lookup.ready():
        raise HTTPException(503, "Index lookup is not ready")
    try:
        vector = lookup.get(payload.index)
        if vector is None:
            raise HTTPException(404, "Unknown index")
        return public_result(request.app.state.predictor.predict_vector(vector, payload.index))
    except HTTPException:
        raise
    except psycopg.Error as exc:
        logger.exception("Lookup database unavailable")
        raise HTTPException(503, "Index lookup is unavailable") from exc
    except Exception as exc:
        logger.exception("Prediction by index failed")
        raise HTTPException(500, "Prediction failed") from exc


@app.post("/api/v1/predict/records")
def predict_json(payload: RecordsRequest, request: Request):
    return {"predictions": predict_records(request, payload.records)}


@app.post("/api/v1/predict/csv")
async def predict_csv(request: Request, file: UploadFile = File(...)):
    if file.content_type not in {"text/csv", "application/csv", "application/vnd.ms-excel", "application/octet-stream"}:
        raise HTTPException(422, "Upload a CSV file")
    raw = await file.read(MAX_CSV_BYTES + 1)
    if len(raw) > MAX_CSV_BYTES:
        raise HTTPException(413, "CSV exceeds 32 MiB")
    try:
        stream = io.StringIO(raw.decode("utf-8-sig"), newline="")
    except UnicodeDecodeError as exc:
        raise HTTPException(422, "CSV must use UTF-8") from exc
    reader = csv.DictReader(stream)
    predictor = request.app.state.predictor
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise HTTPException(422, "CSV needs a header with unique columns")
    expected = predictor.feature_set
    actual = set(reader.fieldnames)
    if expected - actual or actual - expected - {"index", "word_count"}:
        raise HTTPException(422, f"CSV columns differ: missing={sorted(expected - actual)[:5]}, unknown={sorted(actual - expected - {'index', 'word_count'})[:5]}")
    predictions = []
    batch = []
    for number, row in enumerate(reader, start=2):
        if number > MAX_CSV_ROWS + 1:
            raise HTTPException(413, f"CSV exceeds {MAX_CSV_ROWS} rows")
        if None in row or any(value is None for value in row.values()):
            raise HTTPException(422, f"CSV row {number} has the wrong number of columns")
        record = {}
        for key, value in row.items():
            if key in expected:
                try:
                    parsed = None if value.strip() == "" else float(value)
                except ValueError as exc:
                    raise HTTPException(422, f"CSV row {number}: {key} must be numeric") from exc
                record[key] = None if parsed is not None and not math.isfinite(parsed) else parsed
            elif key == "index":
                record[key] = value or None
            else:
                record[key] = value or None
        batch.append(record)
        if len(batch) == 100:
            predictions.extend(predict_records(request, batch))
            batch.clear()
    if batch:
        predictions.extend(predict_records(request, batch))
    if not predictions:
        raise HTTPException(422, "CSV contains no rows")
    return {"predictions": predictions}
