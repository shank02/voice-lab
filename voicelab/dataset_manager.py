"""Dataset Manager: download, cache, subset, decode, and expose Common Voice data."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from voicelab.config import DatasetConfig, settings
from voicelab.errors import DatasetUnavailableError

logger = logging.getLogger(__name__)

DATASET_ID = "cmryvrc6i014vo90786kriix5"
DATASET_VERSION = "26.0"


@dataclass
class Sample:
    """One audio clip with its ground-truth transcript."""

    sample_id: str
    audio: np.ndarray  # float32, mono, 16 kHz
    sampling_rate: int  # always 16000
    reference: str
    duration_s: float


@dataclass
class DatasetInfo:
    """Metadata about the loaded dataset subset."""

    name: str
    version: str
    language: str
    split: str
    num_samples: int
    seed: int
    total_audio_duration_s: float
    sample_ids_hash: str
    cache_path: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _data_root() -> Path:
    return Path(settings.data_dir) / "common_voice"


def _raw_dir() -> Path:
    return _data_root() / "raw"


def _subset_dir() -> Path:
    return _data_root() / "subsets"


def _cache_dir() -> Path:
    return _data_root() / "cache"


def _subset_key(cfg: DatasetConfig) -> str:
    return f"{cfg.split}_{cfg.num_samples}_{cfg.seed}"


def _subset_file(cfg: DatasetConfig) -> Path:
    return _subset_dir() / f"{_subset_key(cfg)}.json"


def _ensure_dataset_downloaded() -> Path:
    """Ensure the raw dataset exists locally. Download via MDC SDK if missing.

    Returns the path to the raw data directory.
    """
    raw = _raw_dir()

    # Check if already downloaded — look for any TSV file
    if raw.exists() and any(raw.rglob("*.tsv")):
        logger.info("Dataset already present at %s", raw)
        return raw

    # Try downloading via Mozilla Data Collective SDK
    if not settings.mdc_api_key:
        raise DatasetUnavailableError(
            f"Dataset not found at {raw} and MDC_API_KEY is not set.\n"
            f"Either:\n"
            f"  1. Set MDC_API_KEY in your .env file and re-run (auto-download), or\n"
            f"  2. Download manually from https://data.mozilla.org and extract to {raw}"
        )

    import os
    os.environ["MDC_API_KEY"] = settings.mdc_api_key

    try:
        import datacollective as dc

        logger.info("Downloading Common Voice Scripted Speech 26.0 via MDC SDK...")
        download_path = dc.download_dataset(
            DATASET_ID,
            download_directory=str(_data_root()),
            show_progress=True,
            overwrite_existing=False,
        )
        logger.info("Downloaded archive to %s", download_path)

        # The SDK downloads a .tar.gz — extract it
        _extract_archive(Path(download_path), raw)

        return raw

    except Exception as e:
        raise DatasetUnavailableError(
            f"Failed to download dataset: {e}\n"
            f"Download manually from https://data.mozilla.org and extract to {raw}"
        ) from e


def _extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Extract a .tar.gz archive to target_dir."""
    import tarfile

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s to %s ...", archive_path, target_dir)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir, filter="data")

    logger.info("Extraction complete.")


def _find_tsv(raw_dir: Path, split: str) -> Path:
    """Find the TSV file for the given split, searching recursively."""
    candidates = list(raw_dir.rglob(f"{split}.tsv"))
    if not candidates:
        # Also try without nesting
        flat = raw_dir / f"{split}.tsv"
        if flat.exists():
            return flat
        raise DatasetUnavailableError(
            f"Could not find {split}.tsv in {raw_dir}. "
            f"Contents: {[str(p.relative_to(raw_dir)) for p in raw_dir.rglob('*') if p.is_file()][:20]}"
        )
    return candidates[0]


def _find_clips_dir(raw_dir: Path) -> Path:
    """Find the directory containing MP3 clips."""
    candidates = list(raw_dir.rglob("clips"))
    if candidates:
        return candidates[0]
    # Maybe clips are directly in raw_dir
    mp3s = list(raw_dir.rglob("*.mp3"))
    if mp3s:
        return mp3s[0].parent
    raise DatasetUnavailableError(f"Could not find audio clips in {raw_dir}")


def _load_tsv(tsv_path: Path) -> pd.DataFrame:
    """Load a Common Voice TSV file."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, na_filter=False)
    return df


def _select_subset(df: pd.DataFrame, cfg: DatasetConfig) -> list[str]:
    """Select a deterministic subset of clip IDs.

    1. Drop rows with empty sentence.
    2. Sort by clip path for deterministic ordering.
    3. Shuffle with seed.
    4. Take first num_samples.
    """
    # Filter empty sentences
    col = "sentence" if "sentence" in df.columns else "sentence_text"
    df = df[df[col].str.strip().astype(bool)].copy()

    # Sort by clip path for deterministic order
    path_col = "path" if "path" in df.columns else "clip"
    df = df.sort_values(path_col).reset_index(drop=True)

    # Shuffle deterministically
    indices = list(range(len(df)))
    rng = random.Random(cfg.seed)
    rng.shuffle(indices)

    # Take first num_samples
    selected_indices = indices[: cfg.num_samples]
    selected = df.iloc[selected_indices]

    return selected[path_col].tolist()


def _compute_sample_ids_hash(sample_ids: list[str]) -> str:
    """SHA256 of sorted sample IDs for reproducibility checks."""
    content = "\n".join(sorted(sample_ids))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _decode_audio(clip_path: Path, target_sr: int) -> tuple[np.ndarray, float]:
    """Decode an audio file to mono float32 at target sample rate.

    Returns (audio_array, duration_seconds).
    """
    audio, sr = librosa.load(str(clip_path), sr=target_sr, mono=True)
    duration = len(audio) / sr
    return audio.astype(np.float32), duration


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_benchmark_samples(cfg: DatasetConfig) -> tuple[list[Sample], DatasetInfo]:
    """Load the benchmark dataset subset.

    Downloads the dataset if needed, selects a deterministic subset,
    decodes and resamples the audio.

    Returns:
        (samples, dataset_info) tuple.
    """
    raw_dir = _ensure_dataset_downloaded()

    # Find the TSV and clips directory
    tsv_path = _find_tsv(raw_dir, cfg.split)
    clips_dir = _find_clips_dir(raw_dir)

    # Load TSV
    df = _load_tsv(tsv_path)
    logger.info("Loaded %d rows from %s", len(df), tsv_path)

    # Determine column names
    path_col = "path" if "path" in df.columns else "clip"
    sent_col = "sentence" if "sentence" in df.columns else "sentence_text"

    # Check for cached subset selection
    subset_file = _subset_file(cfg)
    if subset_file.exists():
        logger.info("Reusing cached subset from %s", subset_file)
        with open(subset_file) as f:
            selected_ids = json.load(f)
    else:
        logger.info("Selecting %d samples with seed=%d", cfg.num_samples, cfg.seed)
        selected_ids = _select_subset(df, cfg)
        subset_file.parent.mkdir(parents=True, exist_ok=True)
        with open(subset_file, "w") as f:
            json.dump(selected_ids, f, indent=2)
        logger.info("Saved subset selection to %s", subset_file)

    # Check for cached decoded audio
    cache_key = _subset_key(cfg)
    cache_file = _cache_dir() / f"{cache_key}.npz"
    use_cache = cache_file.exists()

    cached_audio: dict[str, np.ndarray] = {}
    if use_cache:
        logger.info("Loading cached audio from %s", cache_file)
        with np.load(cache_file) as data:
            cached_audio = {k: data[k] for k in data.files}

    # Build an index from path -> row for fast lookup
    df_indexed = df.set_index(path_col)

    samples: list[Sample] = []
    audio_to_cache: dict[str, np.ndarray] = {}
    total_duration = 0.0

    for clip_id in selected_ids:
        if clip_id not in df_indexed.index:
            logger.warning("Clip %s not found in TSV, skipping", clip_id)
            continue

        row = df_indexed.loc[clip_id]
        sentence = row[sent_col] if isinstance(row, pd.Series) else row.iloc[0][sent_col]

        # Decode or load from cache
        if clip_id in cached_audio:
            audio = cached_audio[clip_id]
            duration = len(audio) / cfg.sampling_rate
        else:
            clip_path = clips_dir / clip_id
            if not clip_path.exists():
                # Try adding .mp3 extension
                clip_path = clips_dir / f"{clip_id}"
                if not clip_path.exists():
                    logger.warning("Audio file not found: %s, skipping", clip_path)
                    continue

            audio, duration = _decode_audio(clip_path, cfg.sampling_rate)
            audio_to_cache[clip_id] = audio

        # Apply duration filter
        if duration > cfg.max_duration_s:
            logger.debug("Skipping %s (%.1fs > %.1fs)", clip_id, duration, cfg.max_duration_s)
            continue

        total_duration += duration
        samples.append(
            Sample(
                sample_id=clip_id,
                audio=audio,
                sampling_rate=cfg.sampling_rate,
                reference=sentence,
                duration_s=duration,
            )
        )

    # Save decoded audio to cache if we had new decodes
    if audio_to_cache and not use_cache:
        _cache_dir().mkdir(parents=True, exist_ok=True)
        all_audio = {**cached_audio, **audio_to_cache}
        np.savez(cache_file, **all_audio)
        logger.info("Cached %d decoded audio arrays to %s", len(all_audio), cache_file)

    sample_ids = [s.sample_id for s in samples]
    info = DatasetInfo(
        name=cfg.name,
        version=DATASET_VERSION,
        language=cfg.language,
        split=cfg.split,
        num_samples=len(samples),
        seed=cfg.seed,
        total_audio_duration_s=total_duration,
        sample_ids_hash=_compute_sample_ids_hash(sample_ids),
        cache_path=str(_data_root()),
    )

    logger.info(
        "Loaded %d samples (%.1fs total audio, hash=%s)",
        len(samples),
        total_duration,
        info.sample_ids_hash,
    )
    return samples, info


def get_dataset_info(cfg: DatasetConfig) -> DatasetInfo:
    """Get dataset metadata without loading audio.

    Returns DatasetInfo. If the subset hasn't been selected yet,
    returns metadata with num_samples=0.
    """
    subset_file = _subset_file(cfg)
    cached = subset_file.exists()

    if cached:
        with open(subset_file) as f:
            selected_ids = json.load(f)
        return DatasetInfo(
            name=cfg.name,
            version=DATASET_VERSION,
            language=cfg.language,
            split=cfg.split,
            num_samples=len(selected_ids),
            seed=cfg.seed,
            total_audio_duration_s=0.0,  # Would need to decode to know
            sample_ids_hash=_compute_sample_ids_hash(selected_ids),
            cache_path=str(_data_root()),
        )

    return DatasetInfo(
        name=cfg.name,
        version=DATASET_VERSION,
        language=cfg.language,
        split=cfg.split,
        num_samples=0,
        seed=cfg.seed,
        total_audio_duration_s=0.0,
        sample_ids_hash="",
        cache_path=str(_data_root()),
    )


def is_dataset_cached() -> bool:
    """Check if the raw dataset has been downloaded."""
    raw = _raw_dir()
    return raw.exists() and any(raw.rglob("*.tsv"))
