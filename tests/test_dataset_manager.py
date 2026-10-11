"""Tests for voicelab.dataset_manager."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from voicelab.config import DatasetConfig
from voicelab.dataset_manager import (
    _compute_sample_ids_hash,
    _decode_audio,
    _load_tsv,
    _select_subset,
    load_benchmark_samples,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_TSV = FIXTURES / "test.tsv"
FIXTURE_CLIPS = FIXTURES / "clips"


# --- TSV loading ---


class TestLoadTSV:
    def test_loads_all_rows(self):
        df = _load_tsv(FIXTURE_TSV)
        assert len(df) == 6  # includes the empty-sentence row

    def test_has_expected_columns(self):
        df = _load_tsv(FIXTURE_TSV)
        assert "path" in df.columns
        assert "sentence" in df.columns


# --- Subset selection ---


class TestSubsetSelection:
    def test_filters_empty_sentences(self):
        df = _load_tsv(FIXTURE_TSV)
        cfg = DatasetConfig(num_samples=10, seed=42)
        selected = _select_subset(df, cfg)
        # clip_006.mp3 has empty sentence — should be excluded
        assert "clip_006.mp3" not in selected

    def test_deterministic_same_seed(self):
        df = _load_tsv(FIXTURE_TSV)
        cfg = DatasetConfig(num_samples=3, seed=42)
        sel1 = _select_subset(df, cfg)
        sel2 = _select_subset(df, cfg)
        assert sel1 == sel2

    def test_different_seed_different_order(self):
        df = _load_tsv(FIXTURE_TSV)
        sel1 = _select_subset(df, DatasetConfig(num_samples=5, seed=42))
        sel2 = _select_subset(df, DatasetConfig(num_samples=5, seed=99))
        # Same items (all 5 valid clips) but different order
        assert set(sel1) == set(sel2)
        assert sel1 != sel2

    def test_respects_num_samples(self):
        df = _load_tsv(FIXTURE_TSV)
        cfg = DatasetConfig(num_samples=2, seed=42)
        selected = _select_subset(df, cfg)
        assert len(selected) == 2


# --- Audio decoding ---


class TestDecodeAudio:
    def test_output_shape_and_dtype(self):
        clip = FIXTURE_CLIPS / "clip_001.mp3"
        audio, duration = _decode_audio(clip, target_sr=16000)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert duration > 0

    def test_resampling(self):
        clip = FIXTURE_CLIPS / "clip_001.mp3"
        audio, duration = _decode_audio(clip, target_sr=8000)
        # At 8 kHz, 1s of audio = 8000 samples
        expected_samples = int(8000 * duration)
        assert abs(len(audio) - expected_samples) < 100

    def test_duration_approximately_correct(self):
        # clip_003 is 2.0 seconds
        clip = FIXTURE_CLIPS / "clip_003.mp3"
        _, duration = _decode_audio(clip, target_sr=16000)
        assert abs(duration - 2.0) < 0.1


# --- Hash ---


class TestSampleIdsHash:
    def test_deterministic(self):
        ids = ["clip_003", "clip_001", "clip_002"]
        h1 = _compute_sample_ids_hash(ids)
        h2 = _compute_sample_ids_hash(ids)
        assert h1 == h2

    def test_order_independent(self):
        h1 = _compute_sample_ids_hash(["a", "b", "c"])
        h2 = _compute_sample_ids_hash(["c", "a", "b"])
        assert h1 == h2  # sorted internally


# --- Integration: load_benchmark_samples with fixtures ---


class TestLoadBenchmarkSamples:
    """Integration test using fixture data with mocked paths."""

    @pytest.fixture(autouse=True)
    def setup_temp_data(self, tmp_path):
        """Set up a temp data dir that mimics the expected structure."""
        self.data_dir = tmp_path / "data" / "common_voice"
        raw_dir = self.data_dir / "raw"
        raw_dir.mkdir(parents=True)

        # Copy fixture TSV and clips
        shutil.copy(FIXTURE_TSV, raw_dir / "test.tsv")
        clips_dst = raw_dir / "clips"
        shutil.copytree(FIXTURE_CLIPS, clips_dst)

        # Patch settings.data_dir
        self._patches = [
            patch("voicelab.dataset_manager.settings.data_dir", str(tmp_path / "data")),
            patch("voicelab.dataset_manager.settings.mdc_api_key", ""),
        ]
        for p in self._patches:
            p.start()

        yield

        for p in self._patches:
            p.stop()

    def test_loads_correct_count(self):
        cfg = DatasetConfig(num_samples=3, seed=42, split="test")
        samples, info = load_benchmark_samples(cfg)
        assert len(samples) == 3
        assert info.num_samples == 3

    def test_samples_have_correct_format(self):
        cfg = DatasetConfig(num_samples=2, seed=42, split="test")
        samples, _ = load_benchmark_samples(cfg)
        for s in samples:
            assert s.audio.dtype == np.float32
            assert s.sampling_rate == 16000
            assert s.duration_s > 0
            assert len(s.reference) > 0
            assert len(s.sample_id) > 0

    def test_deterministic_across_calls(self):
        cfg = DatasetConfig(num_samples=3, seed=42, split="test")
        _, info1 = load_benchmark_samples(cfg)
        _, info2 = load_benchmark_samples(cfg)
        assert info1.sample_ids_hash == info2.sample_ids_hash

    def test_all_clips_same_hash_regardless_of_seed(self):
        # When selecting ALL valid clips, different seeds give the same set
        # (just in different order), so the hash (sorted) should match.
        _, info1 = load_benchmark_samples(DatasetConfig(num_samples=5, seed=42, split="test"))
        _, info2 = load_benchmark_samples(DatasetConfig(num_samples=5, seed=99, split="test"))
        assert info1.sample_ids_hash == info2.sample_ids_hash

    def test_subset_file_persisted(self):
        cfg = DatasetConfig(num_samples=3, seed=42, split="test")
        load_benchmark_samples(cfg)
        subset_file = self.data_dir / "subsets" / "test_3_42.json"
        assert subset_file.exists()
        with open(subset_file) as f:
            ids = json.load(f)
        assert len(ids) == 3

    def test_reuses_cached_subset(self):
        cfg = DatasetConfig(num_samples=3, seed=42, split="test")
        # First call creates subset
        _, info1 = load_benchmark_samples(cfg)
        # Modify the subset file to prove it's reused
        subset_file = self.data_dir / "subsets" / "test_3_42.json"
        with open(subset_file) as f:
            ids = json.load(f)
        # Keep only first 2
        with open(subset_file, "w") as f:
            json.dump(ids[:2], f)
        # Second call reuses the modified subset
        samples, info2 = load_benchmark_samples(cfg)
        assert len(samples) == 2
