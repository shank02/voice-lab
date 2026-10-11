"""Model registry and Transcriber for Whisper inference."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from voicelab.errors import ModelLoadError

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, str] = {
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
}


def list_models() -> list[dict[str, str]]:
    """Return available models as a list of {name, hf_model_id} dicts."""
    return [{"name": k, "hf_model_id": v} for k, v in MODEL_REGISTRY.items()]


class Transcriber:
    """Whisper-based transcriber using Hugging Face Transformers."""

    def __init__(
        self,
        model_name: str,
        device: str,
        language: str = "en",
        task: str = "transcribe",
    ) -> None:
        if model_name not in MODEL_REGISTRY:
            raise ModelLoadError(
                f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        self.hf_model_id = MODEL_REGISTRY[model_name]
        self.device = device
        self.language = language
        self.task = task
        self.model_load_time_s: float = 0.0
        self.hf_model_revision: str = ""

        self._load_model()

    def _load_model(self) -> None:
        """Load processor and model from HF Hub."""
        t0 = time.perf_counter()

        try:
            self.processor = WhisperProcessor.from_pretrained(self.hf_model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.hf_model_id,
                dtype=torch.float32,  # float32 on both CPU and MPS
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise ModelLoadError(f"Failed to load model '{self.model_name}': {e}") from e

        self.model_load_time_s = time.perf_counter() - t0

        # Record model revision for reproducibility
        self.hf_model_revision = getattr(self.model.config, "_commit_hash", "") or ""

        logger.info(
            "Loaded %s on %s in %.1fs (revision: %s)",
            self.model_name,
            self.device,
            self.model_load_time_s,
            self.hf_model_revision[:8],
        )

    def transcribe(self, audio: np.ndarray, sampling_rate: int) -> str:
        """Transcribe a single audio clip.

        Args:
            audio: float32 mono audio array.
            sampling_rate: Sample rate of the audio (should be 16000).

        Returns:
            Predicted transcript string.
        """
        input_features = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=self.language,
                task=self.task,
                do_sample=False,
                num_beams=1,
            )

        # Synchronize MPS before timing ends (caller handles timing)
        if self.device == "mps":
            torch.mps.synchronize()

        transcript = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        return transcript

    def warmup(self) -> None:
        """Run one transcription of silence to warm up the model."""
        silence = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        self.transcribe(silence, sampling_rate=16000)
        logger.info("Model warmup complete.")
