"""Benchmark pipeline: the core orchestration function."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import mlflow

from voicelab.config import BenchmarkConfig
from voicelab.dataset_manager import DatasetInfo, load_benchmark_samples
from voicelab.device import resolve_device
from voicelab.errors import InferenceError, VoiceLabError
from voicelab.evaluation import NORMALIZER_NAME, EvaluationReport, SampleResult, evaluate
from voicelab.models import Transcriber
from voicelab import tracking

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Summary returned after a benchmark run completes."""

    run_id: str
    status: str
    model_name: str
    device: str
    num_samples: int
    wer: float
    cer: float
    avg_latency_s: float
    rtf: float
    total_runtime_s: float


def run_benchmark(cfg: BenchmarkConfig) -> BenchmarkResult:
    """Execute the full benchmark pipeline.

    Dataset Manager → Inference → Evaluation → Experiment Tracking.
    Synchronous — blocks until complete.
    """
    t_start = time.perf_counter()

    # Resolve device
    device = resolve_device(cfg.inference.device)
    cfg.inference.device = device  # record the resolved value

    # Start MLflow run first so failures are recorded
    active_run = tracking.start_run()
    run_id = active_run.info.run_id
    logger.info("Started benchmark run %s", run_id)

    try:
        # --- Stage 1: Dataset ---
        logger.info("Loading dataset...")
        samples, dataset_info = load_benchmark_samples(cfg.dataset)
        logger.info("Loaded %d samples (%.1fs audio)", len(samples), dataset_info.total_audio_duration_s)

        # --- Stage 2: Model Inference ---
        logger.info("Loading model %s on %s...", cfg.model.name, device)
        transcriber = Transcriber(
            model_name=cfg.model.name,
            device=device,
            language=cfg.inference.language,
            task=cfg.inference.task,
        )

        # Log all params once
        tracking.log_params(
            cfg,
            extra={
                "dataset_version": dataset_info.version,
                "sample_ids_hash": dataset_info.sample_ids_hash,
                "normalizer": NORMALIZER_NAME,
                "hf_model_id": transcriber.hf_model_id,
                "hf_model_revision": transcriber.hf_model_revision,
            },
        )

        # Warmup
        transcriber.warmup()

        # Transcribe each sample
        logger.info("Running inference on %d samples...", len(samples))
        sample_results: list[SampleResult] = []
        num_failed = 0

        for i, sample in enumerate(samples):
            try:
                t0 = time.perf_counter()
                prediction = transcriber.transcribe(sample.audio, sample.sampling_rate)
                latency = time.perf_counter() - t0

                sample_results.append(
                    SampleResult(
                        sample_id=sample.sample_id,
                        duration_s=sample.duration_s,
                        reference=sample.reference,
                        prediction=prediction,
                        latency_s=latency,
                    )
                )
            except Exception as e:
                num_failed += 1
                logger.warning("Sample %s failed: %s", sample.sample_id, e)
                sample_results.append(
                    SampleResult(
                        sample_id=sample.sample_id,
                        duration_s=sample.duration_s,
                        reference=sample.reference,
                        prediction="",
                        latency_s=0.0,
                        error=str(e),
                    )
                )

            if (i + 1) % 50 == 0:
                logger.info("Progress: %d/%d samples", i + 1, len(samples))

        # Check failure threshold
        fail_rate = num_failed / len(samples) if samples else 0
        if fail_rate > 0.1:
            raise InferenceError(
                f"{num_failed}/{len(samples)} samples failed ({fail_rate:.0%}), exceeding 10% threshold"
            )

        # --- Stage 3: Evaluation ---
        logger.info("Computing evaluation metrics...")
        report = evaluate(sample_results)

        # --- Stage 4: Tracking ---
        total_runtime = time.perf_counter() - t_start

        tracking.log_metrics(
            report,
            extra={
                "total_runtime_s": total_runtime,
                "model_load_time_s": transcriber.model_load_time_s,
            },
        )
        tracking.log_tags()
        tracking.log_artifacts(
            report=report,
            cfg=cfg,
            dataset_info=asdict(dataset_info),
        )
        tracking.end_run(status="FINISHED")

        result = BenchmarkResult(
            run_id=run_id,
            status="FINISHED",
            model_name=cfg.model.name,
            device=device,
            num_samples=len(samples),
            wer=report.wer,
            cer=report.cer,
            avg_latency_s=report.avg_latency_s,
            rtf=report.rtf,
            total_runtime_s=total_runtime,
        )
        logger.info(
            "Benchmark complete: WER=%.4f, CER=%.4f, RTF=%.3f, runtime=%.1fs",
            result.wer,
            result.cer,
            result.rtf,
            result.total_runtime_s,
        )
        return result

    except VoiceLabError:
        mlflow.set_tag("voicelab.error", str(mlflow.active_run() and "pipeline_error"))
        tracking.end_run(status="FAILED")
        raise

    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        tracking.end_run(status="KILLED")
        raise

    except Exception as e:
        logger.error("Unexpected error: %s", e)
        mlflow.set_tag("voicelab.error", str(e)[:250])
        tracking.end_run(status="FAILED")
        raise VoiceLabError(f"Benchmark failed: {e}") from e
