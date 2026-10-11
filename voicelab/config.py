"""Configuration loading for VoiceLab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env file from project root (if it exists)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# Paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "default.yaml"


# --- Dataclasses matching configs/default.yaml ---


@dataclass
class DatasetConfig:
    name: str = "common_voice_scripted"
    dataset_id: str = "cmryvrc6i014vo90786kriix5"
    language: str = "en-US-south"
    split: str = "test"
    num_samples: int = 200
    seed: int = 42
    sampling_rate: int = 16000
    max_duration_s: float = 30.0


@dataclass
class ModelConfig:
    name: str = "whisper-tiny"


@dataclass
class InferenceConfig:
    device: str = "auto"
    batch_size: int = 1
    language: str = "en"
    task: str = "transcribe"


@dataclass
class BenchmarkConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# --- Environment settings (read from env / .env) ---


@dataclass
class Settings:
    data_dir: str = "./data"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_artifact_root: str = "./mlruns"
    experiment_name: str = "voicelab"
    api_url: str = "http://localhost:8000"
    hf_home: str = "./data/hf"
    mdc_api_key: str = ""


def _load_env_settings() -> Settings:
    """Load settings from environment variables, falling back to defaults."""
    import os

    return Settings(
        data_dir=os.getenv("VOICELAB_DATA_DIR", Settings.data_dir),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", Settings.mlflow_tracking_uri),
        mlflow_artifact_root=os.getenv("MLFLOW_ARTIFACT_ROOT", Settings.mlflow_artifact_root),
        experiment_name=os.getenv("VOICELAB_EXPERIMENT_NAME", Settings.experiment_name),
        api_url=os.getenv("VOICELAB_API_URL", Settings.api_url),
        hf_home=os.getenv("HF_HOME", Settings.hf_home),
        mdc_api_key=os.getenv("MDC_API_KEY", Settings.mdc_api_key),
    )


def load_benchmark_config(overrides: dict | None = None) -> BenchmarkConfig:
    """Load benchmark config from configs/default.yaml, then apply overrides."""
    raw: dict = {}
    if _DEFAULT_CONFIG.exists():
        with open(_DEFAULT_CONFIG) as f:
            raw = yaml.safe_load(f) or {}

    ds = raw.get("dataset", {})
    model = raw.get("model", {})
    inf = raw.get("inference", {})

    cfg = BenchmarkConfig(
        dataset=DatasetConfig(**{k: v for k, v in ds.items() if k in DatasetConfig.__dataclass_fields__}),
        model=ModelConfig(**{k: v for k, v in model.items() if k in ModelConfig.__dataclass_fields__}),
        inference=InferenceConfig(**{k: v for k, v in inf.items() if k in InferenceConfig.__dataclass_fields__}),
    )

    # Apply overrides
    if overrides:
        if "model" in overrides:
            cfg.model.name = overrides["model"]
        if "num_samples" in overrides:
            cfg.dataset.num_samples = overrides["num_samples"]
        if "seed" in overrides:
            cfg.dataset.seed = overrides["seed"]
        if "device" in overrides:
            cfg.inference.device = overrides["device"]

    return cfg


# Module-level singleton
settings = _load_env_settings()
