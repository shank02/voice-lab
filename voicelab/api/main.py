"""FastAPI application for VoiceLab."""

from __future__ import annotations

import csv
import io
import logging
import threading
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query, Response

from voicelab.api.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    CompareResponse,
    DatasetInfoResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    PredictionRow,
    RunDetail,
    RunSummary,
)
from voicelab.benchmark_pipeline import run_benchmark
from voicelab.config import load_benchmark_config
from voicelab.dataset_manager import get_dataset_info, is_dataset_cached
from voicelab.device import resolve_device
from voicelab.errors import (
    DatasetUnavailableError,
    InferenceError,
    ModelLoadError,
    RunNotFoundError,
)
from voicelab.models import MODEL_REGISTRY, list_models
from voicelab import tracking

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VoiceLab API",
    description="Benchmarking platform for speech AI research",
    version="0.1.0",
)

# Lock to prevent concurrent benchmark runs
_benchmark_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Routes — static routes before parameterized ones
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    device = resolve_device("auto")
    return HealthResponse(status="ok", device=device)


@app.get("/models", response_model=list[ModelInfo])
def get_models():
    return [ModelInfo(**m) for m in list_models()]


@app.get("/datasets", response_model=list[DatasetInfoResponse])
def get_datasets():
    cfg = load_benchmark_config()
    info = get_dataset_info(cfg.dataset)
    cached = is_dataset_cached()
    return [
        DatasetInfoResponse(
            **asdict(info),
            cached=cached,
        )
    ]


@app.post("/benchmarks", response_model=BenchmarkResponse)
def create_benchmark(request: BenchmarkRequest):
    # Validate model name
    if request.model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{request.model}'. Available: {list(MODEL_REGISTRY.keys())}",
        )

    # Only one benchmark at a time
    acquired = _benchmark_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=409,
            detail="A benchmark is already running. Please wait for it to complete.",
        )

    try:
        cfg = load_benchmark_config(
            overrides={
                "model": request.model,
                "num_samples": request.num_samples,
                "seed": request.seed,
                "device": request.device,
            }
        )
        result = run_benchmark(cfg)
        return BenchmarkResponse(
            run_id=result.run_id,
            status=result.status,
            model_name=result.model_name,
            device=result.device,
            num_samples=result.num_samples,
            wer=result.wer,
            cer=result.cer,
            avg_latency_s=result.avg_latency_s,
            rtf=result.rtf,
            total_runtime_s=result.total_runtime_s,
        )
    except DatasetUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ModelLoadError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except InferenceError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Benchmark failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _benchmark_lock.release()


# --- Run routes: static paths first ---


@app.get("/runs/compare", response_model=CompareResponse)
def compare_runs(ids: str = Query(description="Comma-separated run IDs")):
    run_ids = [rid.strip() for rid in ids.split(",") if rid.strip()]
    if not run_ids:
        raise HTTPException(status_code=422, detail="No run IDs provided")
    try:
        runs = tracking.compare_runs(run_ids)
        return CompareResponse(runs=[RunDetail(**r) for r in runs])
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/runs/export.csv")
def export_comparison_csv(ids: str = Query(description="Comma-separated run IDs")):
    run_ids = [rid.strip() for rid in ids.split(",") if rid.strip()]
    if not run_ids:
        raise HTTPException(status_code=422, detail="No run IDs provided")

    try:
        runs = tracking.compare_runs(run_ids)
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    output = io.StringIO()
    if runs:
        # Collect all metric and param keys
        metric_keys = sorted({k for r in runs for k in r["metrics"]})
        param_keys = sorted({k for r in runs for k in r["params"]})
        fieldnames = ["run_id", "status"] + param_keys + metric_keys

        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for r in runs:
            row = {"run_id": r["run_id"], "status": r["status"]}
            row.update(r["params"])
            row.update({k: f"{v:.6f}" for k, v in r["metrics"].items()})
            writer.writerow(row)

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=comparison.csv"},
    )


@app.get("/runs", response_model=list[RunSummary])
def list_runs():
    runs = tracking.list_runs()
    return [RunSummary(**r) for r in runs]


# --- Parameterized run routes ---


@app.get("/runs/{run_id}", response_model=RunDetail)
def get_run(run_id: str):
    try:
        run = tracking.get_run(run_id)
        return RunDetail(**run)
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/runs/{run_id}/predictions", response_model=list[PredictionRow])
def get_predictions(run_id: str):
    try:
        predictions = tracking.get_predictions(run_id)
        return [PredictionRow(**p) for p in predictions]
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/runs/{run_id}/export.csv")
def export_predictions_csv(run_id: str):
    try:
        csv_bytes = tracking.get_predictions_csv_bytes(run_id)
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not csv_bytes:
        raise HTTPException(status_code=404, detail="No predictions artifact found")

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{run_id[:8]}.csv"},
    )
