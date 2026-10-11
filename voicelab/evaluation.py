"""Evaluation engine: text normalization and metric computation."""

from __future__ import annotations

from dataclasses import dataclass

import json
import logging

import jiwer
import numpy as np
from huggingface_hub import hf_hub_download
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

logger = logging.getLogger(__name__)

NORMALIZER_NAME = "whisper_english"
_normalizer: EnglishTextNormalizer | None = None


def _get_normalizer() -> EnglishTextNormalizer:
    """Lazily load the Whisper English normalizer with spelling mapping from HF Hub."""
    global _normalizer
    if _normalizer is None:
        mapping_path = hf_hub_download("openai/whisper-tiny", "normalizer.json")
        with open(mapping_path) as f:
            mapping = json.load(f)
        _normalizer = EnglishTextNormalizer(mapping)
    return _normalizer


def normalize(text: str) -> str:
    """Normalize text using Whisper's English normalizer."""
    return _get_normalizer()(text)


@dataclass
class SampleResult:
    """Input to the evaluator: one sample's inference output."""

    sample_id: str
    duration_s: float
    reference: str
    prediction: str
    latency_s: float
    error: str | None = None


@dataclass
class SampleMetrics:
    """Per-sample evaluation output."""

    sample_id: str
    duration_s: float
    reference: str
    prediction: str
    reference_normalized: str
    prediction_normalized: str
    wer: float | None
    cer: float | None
    latency_s: float
    excluded: bool
    error: str | None


@dataclass
class EvaluationReport:
    """Corpus-level metrics plus per-sample details."""

    wer: float
    cer: float
    avg_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    rtf: float
    total_audio_duration_s: float
    num_failed_samples: int
    samples: list[SampleMetrics]


def evaluate(records: list[SampleResult]) -> EvaluationReport:
    """Compute corpus-level and per-sample metrics.

    Args:
        records: One SampleResult per audio clip, in order of processing.

    Returns:
        EvaluationReport with corpus metrics and per-sample breakdowns.
    """
    sample_metrics: list[SampleMetrics] = []
    included_refs: list[str] = []
    included_preds: list[str] = []
    latencies: list[float] = []
    total_duration = 0.0
    num_failed = 0

    for r in records:
        ref_norm = normalize(r.reference)
        pred_norm = normalize(r.prediction)

        # A sample is excluded from WER/CER if its normalized reference is empty
        excluded = len(ref_norm.strip()) == 0

        sample_wer: float | None = None
        sample_cer: float | None = None

        if not excluded:
            included_refs.append(ref_norm)
            included_preds.append(pred_norm)
            # Per-sample metrics for sorting/inspection
            sample_wer = jiwer.wer(ref_norm, pred_norm)
            sample_cer = jiwer.cer(ref_norm, pred_norm)

        if r.error:
            num_failed += 1

        latencies.append(r.latency_s)
        total_duration += r.duration_s

        sample_metrics.append(
            SampleMetrics(
                sample_id=r.sample_id,
                duration_s=r.duration_s,
                reference=r.reference,
                prediction=r.prediction,
                reference_normalized=ref_norm,
                prediction_normalized=pred_norm,
                wer=sample_wer,
                cer=sample_cer,
                latency_s=r.latency_s,
                excluded=excluded,
                error=r.error,
            )
        )

    # Corpus-level WER and CER
    if included_refs:
        corpus_wer = jiwer.wer(included_refs, included_preds)
        corpus_cer = jiwer.cer(included_refs, included_preds)
    else:
        corpus_wer = 0.0
        corpus_cer = 0.0

    # Latency statistics
    lat_array = np.array(latencies) if latencies else np.array([0.0])
    avg_latency = float(np.mean(lat_array))
    p50_latency = float(np.percentile(lat_array, 50))
    p95_latency = float(np.percentile(lat_array, 95))

    # RTF
    total_inference_time = float(np.sum(lat_array))
    rtf = total_inference_time / total_duration if total_duration > 0 else 0.0

    return EvaluationReport(
        wer=corpus_wer,
        cer=corpus_cer,
        avg_latency_s=avg_latency,
        p50_latency_s=p50_latency,
        p95_latency_s=p95_latency,
        rtf=rtf,
        total_audio_duration_s=total_duration,
        num_failed_samples=num_failed,
        samples=sample_metrics,
    )
