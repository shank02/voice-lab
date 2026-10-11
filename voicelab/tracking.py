"""MLflow experiment tracking: write and read helpers.

This is the only module that imports mlflow.
"""

from __future__ import annotations

import csv
import io
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import torch
import transformers
import yaml

import voicelab
from voicelab.config import BenchmarkConfig, settings
from voicelab.errors import RunNotFoundError
from voicelab.evaluation import EvaluationReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _ensure_experiment() -> str:
    """Set up MLflow tracking URI and experiment. Returns experiment ID."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(settings.experiment_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(
            settings.experiment_name,
            artifact_location=settings.mlflow_artifact_root,
        )
    else:
        exp_id = experiment.experiment_id
    return exp_id


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def start_run() -> mlflow.ActiveRun:
    """Start a new MLflow run in the voicelab experiment."""
    exp_id = _ensure_experiment()
    return mlflow.start_run(experiment_id=exp_id)


def log_params(cfg: BenchmarkConfig, extra: dict[str, Any] | None = None) -> None:
    """Log benchmark configuration as MLflow params."""
    params = {
        "model_name": cfg.model.name,
        "device": cfg.inference.device,
        "dataset_name": cfg.dataset.name,
        "language": cfg.dataset.language,
        "split": cfg.dataset.split,
        "num_samples": cfg.dataset.num_samples,
        "seed": cfg.dataset.seed,
        "batch_size": cfg.inference.batch_size,
    }
    if extra:
        params.update(extra)
    mlflow.log_params(params)


def log_metrics(report: EvaluationReport, extra: dict[str, float] | None = None) -> None:
    """Log evaluation metrics."""
    metrics = {
        "wer": report.wer,
        "cer": report.cer,
        "avg_latency_s": report.avg_latency_s,
        "p50_latency_s": report.p50_latency_s,
        "p95_latency_s": report.p95_latency_s,
        "rtf": report.rtf,
        "total_audio_duration_s": report.total_audio_duration_s,
        "num_failed_samples": report.num_failed_samples,
    }
    if extra:
        metrics.update(extra)
    mlflow.log_metrics(metrics)


def log_tags() -> None:
    """Log environment tags for reproducibility."""
    mlflow.set_tags(
        {
            "voicelab.version": voicelab.__version__,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "platform": platform.platform(),
        }
    )


def log_artifacts(
    report: EvaluationReport,
    cfg: BenchmarkConfig,
    dataset_info: dict[str, Any],
) -> None:
    """Write and log predictions.csv, config.yaml, and dataset_info.json."""
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # predictions.csv
        csv_path = tmpdir / "predictions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "duration_s",
                    "reference",
                    "prediction",
                    "reference_normalized",
                    "prediction_normalized",
                    "wer",
                    "cer",
                    "latency_s",
                    "excluded",
                    "error",
                ],
            )
            writer.writeheader()
            for s in report.samples:
                writer.writerow(
                    {
                        "sample_id": s.sample_id,
                        "duration_s": f"{s.duration_s:.3f}",
                        "reference": s.reference,
                        "prediction": s.prediction,
                        "reference_normalized": s.reference_normalized,
                        "prediction_normalized": s.prediction_normalized,
                        "wer": f"{s.wer:.4f}" if s.wer is not None else "",
                        "cer": f"{s.cer:.4f}" if s.cer is not None else "",
                        "latency_s": f"{s.latency_s:.4f}",
                        "excluded": str(s.excluded),
                        "error": s.error or "",
                    }
                )
        mlflow.log_artifact(str(csv_path))

        # config.yaml
        config_path = tmpdir / "config.yaml"
        cfg_dict = {
            "dataset": asdict(cfg.dataset),
            "model": asdict(cfg.model),
            "inference": asdict(cfg.inference),
        }
        with open(config_path, "w") as f:
            yaml.dump(cfg_dict, f, default_flow_style=False)
        mlflow.log_artifact(str(config_path))

        # dataset_info.json
        info_path = tmpdir / "dataset_info.json"
        with open(info_path, "w") as f:
            json.dump(dataset_info, f, indent=2)
        mlflow.log_artifact(str(info_path))


def end_run(status: str = "FINISHED") -> None:
    """End the current MLflow run with the given status."""
    mlflow.end_run(status=status)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def list_runs() -> list[dict[str, Any]]:
    """List all runs in the voicelab experiment, newest first."""
    exp_id = _ensure_experiment()
    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        order_by=["start_time DESC"],
        output_format="list",
    )
    results = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        results.append(
            {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "model_name": params.get("model_name", ""),
                "device": params.get("device", ""),
                "num_samples": params.get("num_samples", ""),
                "wer": metrics.get("wer"),
                "cer": metrics.get("cer"),
                "rtf": metrics.get("rtf"),
                "avg_latency_s": metrics.get("avg_latency_s"),
                "total_runtime_s": metrics.get("total_runtime_s"),
            }
        )
    return results


def get_run(run_id: str) -> dict[str, Any]:
    """Get full details for a single run."""
    try:
        run = mlflow.get_run(run_id)
    except Exception as e:
        raise RunNotFoundError(f"Run '{run_id}' not found: {e}") from e

    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
    }


def get_predictions(run_id: str) -> list[dict[str, str]]:
    """Read predictions.csv artifact for a run."""
    try:
        run = mlflow.get_run(run_id)
    except Exception as e:
        raise RunNotFoundError(f"Run '{run_id}' not found: {e}") from e

    artifact_uri = run.info.artifact_uri
    csv_path = Path(artifact_uri) / "predictions.csv"

    # Handle file:// URI scheme
    path_str = str(csv_path)
    if path_str.startswith("file://"):
        path_str = path_str[7:]
    csv_path = Path(path_str)

    if not csv_path.exists():
        return []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_predictions_csv_bytes(run_id: str) -> bytes:
    """Read raw predictions.csv bytes for a run (for download)."""
    try:
        run = mlflow.get_run(run_id)
    except Exception as e:
        raise RunNotFoundError(f"Run '{run_id}' not found: {e}") from e

    artifact_uri = run.info.artifact_uri
    csv_path = Path(artifact_uri) / "predictions.csv"

    path_str = str(csv_path)
    if path_str.startswith("file://"):
        path_str = path_str[7:]
    csv_path = Path(path_str)

    if not csv_path.exists():
        return b""

    return csv_path.read_bytes()


def compare_runs(run_ids: list[str]) -> list[dict[str, Any]]:
    """Get details for multiple runs for side-by-side comparison."""
    return [get_run(rid) for rid in run_ids]
