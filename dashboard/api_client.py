"""Thin requests wrapper around the VoiceLab REST API."""

from __future__ import annotations

import os
from typing import Any

import requests

API_URL = os.getenv("VOICELAB_API_URL", "http://localhost:8000")

# 30-minute timeout for benchmark runs (synchronous, can take a while)
_BENCHMARK_TIMEOUT = 1800
_DEFAULT_TIMEOUT = 30


def _url(path: str) -> str:
    return f"{API_URL}{path}"


def health() -> dict[str, Any]:
    resp = requests.get(_url("/health"), timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def list_models() -> list[dict[str, str]]:
    resp = requests.get(_url("/models"), timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def list_datasets() -> list[dict[str, Any]]:
    resp = requests.get(_url("/datasets"), timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def run_benchmark(
    model: str,
    num_samples: int = 200,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, Any]:
    resp = requests.post(
        _url("/benchmarks"),
        json={"model": model, "num_samples": num_samples, "seed": seed, "device": device},
        timeout=_BENCHMARK_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def list_runs() -> list[dict[str, Any]]:
    resp = requests.get(_url("/runs"), timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_run(run_id: str) -> dict[str, Any]:
    resp = requests.get(_url(f"/runs/{run_id}"), timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_predictions(run_id: str) -> list[dict[str, str]]:
    resp = requests.get(_url(f"/runs/{run_id}/predictions"), timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_predictions_csv(run_id: str) -> bytes:
    resp = requests.get(_url(f"/runs/{run_id}/export.csv"), timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.content


def compare_runs(run_ids: list[str]) -> dict[str, Any]:
    ids_str = ",".join(run_ids)
    resp = requests.get(_url(f"/runs/compare?ids={ids_str}"), timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_comparison_csv(run_ids: list[str]) -> bytes:
    ids_str = ",".join(run_ids)
    resp = requests.get(_url(f"/runs/export.csv?ids={ids_str}"), timeout=_DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return resp.content
