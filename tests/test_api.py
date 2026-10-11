"""Tests for the FastAPI routes."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from voicelab.api.main import app, _benchmark_lock
from voicelab.benchmark_pipeline import BenchmarkResult
from voicelab.errors import DatasetUnavailableError, ModelLoadError, RunNotFoundError

FIXTURES = Path(__file__).parent / "fixtures"

client = TestClient(app)


@pytest.fixture(autouse=True)
def patch_env(tmp_path):
    """Patch settings for all tests."""
    import mlflow

    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    data_dir = tmp_path / "data"
    cv_dir = data_dir / "common_voice" / "raw"
    cv_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "test.tsv", cv_dir / "test.tsv")
    shutil.copytree(FIXTURES / "clips", cv_dir / "clips")

    patches = [
        patch("voicelab.dataset_manager.settings.data_dir", str(data_dir)),
        patch("voicelab.dataset_manager.settings.mdc_api_key", ""),
        patch("voicelab.tracking.settings.mlflow_tracking_uri", mlflow_uri),
        patch("voicelab.tracking.settings.mlflow_artifact_root", str(data_dir / "mlruns")),
        patch("voicelab.tracking.settings.experiment_name", "test_api"),
    ]
    for p in patches:
        p.start()
    yield
    mlflow.end_run()
    for p in patches:
        p.stop()


def _fake_benchmark_result(**overrides) -> BenchmarkResult:
    defaults = dict(
        run_id="abc123",
        status="FINISHED",
        model_name="whisper-tiny",
        device="cpu",
        num_samples=3,
        wer=0.15,
        cer=0.05,
        avg_latency_s=0.5,
        rtf=0.3,
        total_runtime_s=10.0,
    )
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


class TestHealth:
    def test_health(self):
        with patch("voicelab.api.main.resolve_device", return_value="cpu"):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["device"] == "cpu"


class TestModels:
    def test_list_models(self):
        resp = client.get("/models")
        assert resp.status_code == 200
        models = resp.json()
        names = [m["name"] for m in models]
        assert "whisper-tiny" in names
        assert "whisper-base" in names
        assert "whisper-small" in names


class TestDatasets:
    def test_list_datasets(self):
        resp = client.get("/datasets")
        assert resp.status_code == 200
        datasets = resp.json()
        assert len(datasets) == 1
        assert datasets[0]["name"] == "common_voice_scripted"
        assert "cached" in datasets[0]


class TestBenchmarks:
    def test_unknown_model_422(self):
        resp = client.post("/benchmarks", json={"model": "whisper-nonexistent"})
        assert resp.status_code == 422

    def test_successful_benchmark(self):
        result = _fake_benchmark_result()
        with patch("voicelab.api.main.run_benchmark", return_value=result):
            resp = client.post("/benchmarks", json={"model": "whisper-tiny"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == "abc123"
        assert data["status"] == "FINISHED"
        assert data["wer"] == 0.15

    def test_dataset_unavailable_503(self):
        with patch(
            "voicelab.api.main.run_benchmark",
            side_effect=DatasetUnavailableError("not found"),
        ):
            resp = client.post("/benchmarks", json={"model": "whisper-tiny"})
        assert resp.status_code == 503

    def test_model_load_error_503(self):
        with patch(
            "voicelab.api.main.run_benchmark",
            side_effect=ModelLoadError("failed"),
        ):
            resp = client.post("/benchmarks", json={"model": "whisper-tiny"})
        assert resp.status_code == 503

    def test_concurrent_run_409(self):
        """Second benchmark while one is running returns 409."""
        # Acquire the lock to simulate a running benchmark
        _benchmark_lock.acquire()
        try:
            resp = client.post("/benchmarks", json={"model": "whisper-tiny"})
            assert resp.status_code == 409
            assert "already running" in resp.json()["detail"]
        finally:
            _benchmark_lock.release()


class TestRuns:
    def _create_run(self):
        """Helper: run a real benchmark with fake transcriber to populate MLflow."""
        from voicelab.benchmark_pipeline import run_benchmark as real_run
        from voicelab.config import BenchmarkConfig, DatasetConfig, InferenceConfig, ModelConfig

        class FakeTranscriber:
            model_name = "whisper-tiny"
            hf_model_id = "openai/whisper-tiny"
            hf_model_revision = "abc123"
            model_load_time_s = 0.1

            def transcribe(self, audio, sr):
                return "fake"

            def warmup(self):
                pass

        cfg = BenchmarkConfig(
            dataset=DatasetConfig(num_samples=3, seed=42, split="test"),
            model=ModelConfig(name="whisper-tiny"),
            inference=InferenceConfig(device="cpu"),
        )
        with (
            patch("voicelab.benchmark_pipeline.Transcriber", return_value=FakeTranscriber()),
            patch("voicelab.benchmark_pipeline.resolve_device", return_value="cpu"),
        ):
            return real_run(cfg)

    def test_list_runs(self):
        result = self._create_run()
        resp = client.get("/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert any(r["run_id"] == result.run_id for r in runs)

    def test_get_run(self):
        result = self._create_run()
        resp = client.get(f"/runs/{result.run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == result.run_id
        assert "params" in data
        assert "metrics" in data

    def test_get_run_not_found(self):
        resp = client.get("/runs/nonexistent_run_id")
        assert resp.status_code == 404

    def test_get_predictions(self):
        result = self._create_run()
        resp = client.get(f"/runs/{result.run_id}/predictions")
        assert resp.status_code == 200
        predictions = resp.json()
        assert len(predictions) == 3
        assert "sample_id" in predictions[0]

    def test_export_predictions_csv(self):
        result = self._create_run()
        resp = client.get(f"/runs/{result.run_id}/export.csv")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "sample_id" in resp.text

    def test_compare_runs(self):
        r1 = self._create_run()
        r2 = self._create_run()
        resp = client.get(f"/runs/compare?ids={r1.run_id},{r2.run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["runs"]) == 2

    def test_export_comparison_csv(self):
        r1 = self._create_run()
        r2 = self._create_run()
        resp = client.get(f"/runs/export.csv?ids={r1.run_id},{r2.run_id}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "run_id" in resp.text
        # Should have header + 2 data rows
        lines = [l for l in resp.text.strip().split("\n") if l]
        assert len(lines) == 3
