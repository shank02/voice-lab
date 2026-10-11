"""Pydantic request/response schemas for the VoiceLab API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# --- Requests ---


class BenchmarkRequest(BaseModel):
    model: str = Field(description="Model name from the registry (e.g. whisper-tiny)")
    num_samples: int = Field(default=200, ge=1, le=10000)
    seed: int = Field(default=42)
    device: str = Field(default="auto", pattern="^(auto|mps|cpu)$")


# --- Responses ---


class HealthResponse(BaseModel):
    status: str
    device: str


class ModelInfo(BaseModel):
    name: str
    hf_model_id: str


class DatasetInfoResponse(BaseModel):
    name: str
    version: str
    language: str
    split: str
    num_samples: int
    seed: int
    total_audio_duration_s: float
    sample_ids_hash: str
    cache_path: str
    cached: bool


class BenchmarkResponse(BaseModel):
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


class RunSummary(BaseModel):
    run_id: str
    status: str
    start_time: int | None = None
    model_name: str = ""
    device: str = ""
    num_samples: str = ""
    wer: float | None = None
    cer: float | None = None
    rtf: float | None = None
    avg_latency_s: float | None = None
    total_runtime_s: float | None = None


class RunDetail(BaseModel):
    run_id: str
    status: str
    start_time: int | None = None
    end_time: int | None = None
    params: dict[str, str] = {}
    metrics: dict[str, float] = {}
    tags: dict[str, str] = {}


class PredictionRow(BaseModel):
    sample_id: str = ""
    duration_s: str = ""
    reference: str = ""
    prediction: str = ""
    reference_normalized: str = ""
    prediction_normalized: str = ""
    wer: str = ""
    cer: str = ""
    latency_s: str = ""
    excluded: str = ""
    error: str = ""


class CompareResponse(BaseModel):
    runs: list[RunDetail]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    run_id: str | None = None
