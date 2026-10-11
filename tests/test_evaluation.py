"""Tests for voicelab.evaluation."""

import pytest

from voicelab.evaluation import SampleResult, evaluate, normalize


# --- Normalizer tests ---


class TestNormalize:
    def test_lowercases(self):
        assert normalize("HELLO WORLD") == "hello world"

    def test_removes_punctuation(self):
        result = normalize("Hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_empty_string(self):
        assert normalize("") == ""

    def test_whitespace_only(self):
        assert normalize("   ") == ""

    def test_numbers(self):
        # Whisper normalizer expands numbers
        result = normalize("I have 2 cats")
        assert "two" in result or "2" in result  # depends on normalizer version


# --- Helper to build SampleResults ---


def _make_result(
    sample_id: str = "s1",
    duration_s: float = 2.0,
    reference: str = "hello world",
    prediction: str = "hello world",
    latency_s: float = 0.5,
    error: str | None = None,
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        duration_s=duration_s,
        reference=reference,
        prediction=prediction,
        latency_s=latency_s,
        error=error,
    )


# --- WER / CER tests ---


class TestEvaluateWER:
    def test_perfect_transcription(self):
        records = [_make_result(reference="hello world", prediction="hello world")]
        report = evaluate(records)
        assert report.wer == 0.0
        assert report.cer == 0.0

    def test_completely_wrong(self):
        # "hello world" (2 words) vs "foo bar baz" (3 words) => S=2, I=1 => WER = 3/2
        records = [_make_result(reference="hello world", prediction="foo bar baz")]
        report = evaluate(records)
        assert report.wer > 0.0

    def test_one_substitution(self):
        # "the cat sat" vs "the dog sat" => 1 sub out of 3 words => WER ≈ 0.333
        records = [_make_result(reference="the cat sat", prediction="the dog sat")]
        report = evaluate(records)
        assert abs(report.wer - 1.0 / 3.0) < 0.01

    def test_corpus_level_not_mean(self):
        """WER is computed over the full corpus, not averaged per sample."""
        # Sample 1: "a" vs "a" => 0/1
        # Sample 2: "a b c d" vs "x y z w" => 4/4
        # Corpus: 4 errors / 5 words = 0.8
        # Mean of per-sample: (0 + 1) / 2 = 0.5
        records = [
            _make_result(sample_id="s1", reference="a", prediction="a"),
            _make_result(sample_id="s2", reference="a b c d", prediction="x y z w"),
        ]
        report = evaluate(records)
        assert abs(report.wer - 0.8) < 0.01

    def test_empty_prediction_scores_as_deletions(self):
        records = [_make_result(reference="hello world", prediction="")]
        report = evaluate(records)
        assert report.wer == 1.0  # 2 deletions / 2 words

    def test_cer_basic(self):
        # "ab" vs "ac" => 1 sub / 2 chars = 0.5
        records = [_make_result(reference="ab", prediction="ac")]
        report = evaluate(records)
        assert abs(report.cer - 0.5) < 0.01


# --- Exclusion tests ---


class TestExclusion:
    def test_empty_reference_excluded(self):
        records = [_make_result(reference="", prediction="something")]
        report = evaluate(records)
        assert report.samples[0].excluded is True
        assert report.samples[0].wer is None

    def test_whitespace_reference_excluded(self):
        records = [_make_result(reference="   ", prediction="something")]
        report = evaluate(records)
        assert report.samples[0].excluded is True

    def test_mixed_excluded_and_included(self):
        records = [
            _make_result(sample_id="s1", reference="hello", prediction="hello"),
            _make_result(sample_id="s2", reference="", prediction="noise"),
        ]
        report = evaluate(records)
        assert report.wer == 0.0  # only s1 counts
        assert report.samples[1].excluded is True


# --- Latency / RTF tests ---


class TestLatencyMetrics:
    def test_avg_latency(self):
        records = [
            _make_result(sample_id="s1", latency_s=1.0),
            _make_result(sample_id="s2", latency_s=3.0),
        ]
        report = evaluate(records)
        assert abs(report.avg_latency_s - 2.0) < 0.001

    def test_p50_p95(self):
        records = [
            _make_result(sample_id=f"s{i}", latency_s=float(i))
            for i in range(1, 101)
        ]
        report = evaluate(records)
        assert abs(report.p50_latency_s - 50.5) < 1.0
        assert report.p95_latency_s > 90.0

    def test_rtf(self):
        # 2 samples, each 5s audio, each 1s inference => RTF = 2/10 = 0.2
        records = [
            _make_result(sample_id="s1", duration_s=5.0, latency_s=1.0),
            _make_result(sample_id="s2", duration_s=5.0, latency_s=1.0),
        ]
        report = evaluate(records)
        assert abs(report.rtf - 0.2) < 0.001

    def test_total_audio_duration(self):
        records = [
            _make_result(sample_id="s1", duration_s=3.0),
            _make_result(sample_id="s2", duration_s=7.0),
        ]
        report = evaluate(records)
        assert abs(report.total_audio_duration_s - 10.0) < 0.001


# --- Error counting ---


class TestErrorCounting:
    def test_failed_samples_counted(self):
        records = [
            _make_result(sample_id="s1", error=None),
            _make_result(sample_id="s2", error="CUDA OOM"),
            _make_result(sample_id="s3", error="timeout"),
        ]
        report = evaluate(records)
        assert report.num_failed_samples == 2

    def test_no_errors(self):
        records = [_make_result()]
        report = evaluate(records)
        assert report.num_failed_samples == 0
