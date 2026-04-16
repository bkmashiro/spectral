"""Tests for new spectral features: window functions, harmonics, confidence, z-scores, export."""

import json
import math
import tempfile
import os

from spectral.analysis import (
    analyze_signal,
    detect_anomalies,
    apply_window,
    WINDOW_FUNCTIONS,
    _find_harmonics,
    _bootstrap_period_ci,
    compute_summary,
    SpectralPeak,
    Anomaly,
)
from spectral.signal import Signal
from spectral.visualize import render_summary_stats


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

class TestWindowFunctions:
    def test_rectangular_is_all_ones(self):
        w = WINDOW_FUNCTIONS["rectangular"](8)
        assert all(v == 1.0 for v in w)

    def test_hamming_endpoints(self):
        w = WINDOW_FUNCTIONS["hamming"](64)
        # Hamming endpoints are 0.08, not 0
        assert abs(w[0] - 0.08) < 0.01
        assert abs(w[-1] - 0.08) < 0.01

    def test_hanning_endpoints(self):
        w = WINDOW_FUNCTIONS["hanning"](64)
        # Hanning endpoints are 0
        assert abs(w[0]) < 1e-10
        assert abs(w[-1]) < 1e-10

    def test_blackman_endpoints(self):
        w = WINDOW_FUNCTIONS["blackman"](64)
        # Blackman endpoints are ~0
        assert abs(w[0]) < 0.01
        assert abs(w[-1]) < 0.01

    def test_window_lengths(self):
        for name, func in WINDOW_FUNCTIONS.items():
            assert len(func(32)) == 32
            assert len(func(1)) == 1
            assert len(func(0)) == 0

    def test_apply_window_scales(self):
        vals = [1.0] * 8
        result = apply_window(vals, "hamming")
        assert len(result) == 8
        # Interior values should be close to 1, endpoints < 1
        assert result[4] > result[0]

    def test_apply_window_unknown_falls_back(self):
        """Unknown window name should default to hamming."""
        vals = [1.0] * 8
        a = apply_window(vals, "nonexistent")
        b = apply_window(vals, "hamming")
        assert a == b

    def test_window_symmetry(self):
        for name, func in WINDOW_FUNCTIONS.items():
            w = func(32)
            for i in range(len(w)):
                assert abs(w[i] - w[len(w) - 1 - i]) < 1e-10, f"{name} not symmetric at {i}"


class TestWindowedAnalysis:
    def test_window_name_in_result(self):
        sig = Signal(name="t", bucket_size=1.0, start_time=0,
                     values=[math.sin(i * 0.1) for i in range(256)])
        for wname in ["hamming", "hanning", "blackman", "rectangular"]:
            result = analyze_signal(sig, window_name=wname)
            assert result.window_function == wname

    def test_detects_period_with_each_window(self):
        """All windows should detect a clear sine."""
        bucket_size = 10.0
        period = 600.0
        n = 1024
        values = [10.0 + 5.0 * math.sin(2 * math.pi * i * bucket_size / period)
                  for i in range(n)]
        sig = Signal(name="sine", bucket_size=bucket_size, start_time=0, values=values)

        for wname in WINDOW_FUNCTIONS:
            result = analyze_signal(sig, window_name=wname)
            assert len(result.peaks) > 0, f"No peaks with {wname}"
            assert abs(result.peaks[0].period - period) / period < 0.1, \
                f"{wname}: got {result.peaks[0].period}, expected ~{period}"


# ---------------------------------------------------------------------------
# Harmonic detection
# ---------------------------------------------------------------------------

class TestHarmonicDetection:
    def test_finds_sub_harmonic(self):
        """If we have peaks at P and P/2, the harmonic should be detected."""
        peaks = [
            SpectralPeak(frequency=1/600, period=600.0, magnitude=10.0, phase=0.0, rank=1),
            SpectralPeak(frequency=1/300, period=300.0, magnitude=5.0, phase=0.0, rank=2),
        ]
        harmonics = _find_harmonics(600.0, peaks)
        assert 300.0 in harmonics

    def test_finds_super_harmonic(self):
        """If we have peaks at P and 2P, the harmonic should be detected."""
        peaks = [
            SpectralPeak(frequency=1/300, period=300.0, magnitude=10.0, phase=0.0, rank=1),
            SpectralPeak(frequency=1/600, period=600.0, magnitude=5.0, phase=0.0, rank=2),
        ]
        harmonics = _find_harmonics(300.0, peaks)
        assert 600.0 in harmonics

    def test_no_harmonics_when_none(self):
        peaks = [
            SpectralPeak(frequency=1/600, period=600.0, magnitude=10.0, phase=0.0, rank=1),
            SpectralPeak(frequency=1/1500, period=1500.0, magnitude=5.0, phase=0.0, rank=2),
        ]
        harmonics = _find_harmonics(600.0, peaks)
        # 1500 is not a harmonic of 600
        assert 1500.0 not in harmonics

    def test_harmonics_in_analyze(self):
        """analyze_signal with detect_harmonics=True should populate harmonics list."""
        bucket_size = 5.0
        n = 2048
        period = 300.0
        # Signal with fundamental + second harmonic
        values = [
            5.0 * math.sin(2 * math.pi * i * bucket_size / period) +
            2.0 * math.sin(2 * math.pi * i * bucket_size / (period / 2))
            for i in range(n)
        ]
        sig = Signal(name="harm", bucket_size=bucket_size, start_time=0, values=values)
        result = analyze_signal(sig, detect_harmonics=True)
        # The dominant peak should have harmonics detected
        assert len(result.peaks) >= 2
        # At least one peak should have non-empty harmonics
        any_harmonics = any(len(p.harmonics) > 0 for p in result.peaks)
        assert any_harmonics, "No harmonics detected for signal with known harmonics"


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_ci_contains_true_period(self):
        """CI should bracket the true period for a clear signal."""
        bucket_size = 10.0
        period = 600.0
        n = 256
        values = [10.0 + 8.0 * math.sin(2 * math.pi * i * bucket_size / period)
                  for i in range(n)]
        sample_rate = 1.0 / bucket_size
        lo, hi = _bootstrap_period_ci(values, sample_rate, period, n_bootstrap=50)
        assert lo <= period <= hi, f"CI [{lo}, {hi}] does not contain true period {period}"

    def test_ci_narrow_for_strong_signal(self):
        """Strong signal should give narrow CI."""
        bucket_size = 10.0
        period = 600.0
        n = 512
        values = [10.0 + 20.0 * math.sin(2 * math.pi * i * bucket_size / period)
                  for i in range(n)]
        sample_rate = 1.0 / bucket_size
        lo, hi = _bootstrap_period_ci(values, sample_rate, period, n_bootstrap=50)
        width = hi - lo
        assert width < period * 2.0, f"CI too wide: [{lo}, {hi}]"

    def test_ci_short_signal(self):
        """Very short signal should return target period as both bounds."""
        lo, hi = _bootstrap_period_ci([1.0] * 8, 1.0, 100.0, n_bootstrap=10)
        assert lo == 100.0
        assert hi == 100.0

    def test_ci_in_analyze_signal(self):
        """compute_confidence=True should populate confidence fields."""
        bucket_size = 10.0
        period = 600.0
        n = 256
        values = [10.0 + 5.0 * math.sin(2 * math.pi * i * bucket_size / period)
                  for i in range(n)]
        sig = Signal(name="ci_test", bucket_size=bucket_size, start_time=0, values=values)
        result = analyze_signal(sig, compute_confidence=True)
        if result.peaks:
            p = result.peaks[0]
            assert p.confidence_low is not None
            assert p.confidence_high is not None
            assert p.confidence_low <= p.confidence_high


# ---------------------------------------------------------------------------
# Z-score anomaly detection
# ---------------------------------------------------------------------------

class TestZScoreAnomaly:
    def test_z_score_populated(self):
        """Anomalies should have z_score field."""
        n = 512
        values = [1.0] * n
        for i in range(200, 264):
            values[i] = 20.0
        sig = Signal(name="spike", bucket_size=10.0, start_time=0, values=values)
        anomalies = detect_anomalies(sig, window_buckets=64, step_buckets=16, threshold=2.0)
        assert len(anomalies) > 0
        for a in anomalies:
            assert hasattr(a, "z_score")
            assert isinstance(a.z_score, float)

    def test_z_score_high_for_spike(self):
        """Z-score should be high for anomalous windows."""
        n = 512
        values = [1.0] * n
        for i in range(200, 264):
            values[i] = 50.0
        sig = Signal(name="big_spike", bucket_size=10.0, start_time=0, values=values)
        anomalies = detect_anomalies(sig, window_buckets=64, step_buckets=16, threshold=2.0)
        high_z = [a for a in anomalies if a.z_score > 2.0]
        assert len(high_z) > 0

    def test_z_score_threshold_detection(self):
        """Setting z_score_threshold should flag anomalies by z-score alone."""
        n = 512
        values = [1.0] * n
        for i in range(200, 264):
            values[i] = 15.0
        sig = Signal(name="moderate", bucket_size=10.0, start_time=0, values=values)
        # Very high energy threshold, low z threshold
        anomalies = detect_anomalies(
            sig, window_buckets=64, step_buckets=16,
            threshold=100.0, z_score_threshold=1.5,
        )
        # Should still find anomalies via z-score even though energy ratio threshold is high
        assert len(anomalies) > 0


# ---------------------------------------------------------------------------
# JSON export via to_dict()
# ---------------------------------------------------------------------------

class TestExport:
    def test_to_dict_structure(self):
        sig = Signal(name="export_test", bucket_size=1.0, start_time=0,
                     values=[math.sin(i * 0.1) for i in range(256)])
        result = analyze_signal(sig)
        d = result.to_dict()
        assert d["signal_name"] == "export_test"
        assert "frequencies" in d
        assert "magnitudes" in d
        assert "peaks" in d
        assert isinstance(d["peaks"], list)
        assert d["window_function"] == "hamming"

    def test_to_dict_serializable(self):
        """to_dict() output should be JSON-serializable."""
        sig = Signal(name="json_test", bucket_size=1.0, start_time=0,
                     values=[math.sin(i * 0.1) for i in range(128)])
        result = analyze_signal(sig, compute_confidence=True, detect_harmonics=True)
        d = result.to_dict()
        text = json.dumps(d)
        parsed = json.loads(text)
        assert parsed["signal_name"] == "json_test"

    def test_to_dict_peaks_have_ci_fields(self):
        sig = Signal(name="ci_export", bucket_size=10.0, start_time=0,
                     values=[10.0 + 5.0 * math.sin(2 * math.pi * i * 10.0 / 600)
                             for i in range(256)])
        result = analyze_signal(sig, compute_confidence=True)
        d = result.to_dict()
        if d["peaks"]:
            p = d["peaks"][0]
            assert "confidence_low" in p
            assert "confidence_high" in p


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

class TestSummaryStats:
    def test_compute_summary(self):
        values = [float(i % 10) for i in range(100)]
        sig = Signal(name="summary", bucket_size=10.0, start_time=0, values=values)
        s = compute_summary(sig)
        assert s.total_events == sum(int(v) for v in values)
        assert s.time_span_seconds == sig.duration
        assert s.bucket_size == 10.0
        assert s.peak_rate >= s.quietest_rate

    def test_render_summary_stats(self):
        values = [5.0] * 100
        sig = Signal(name="flat", bucket_size=10.0, start_time=0, values=values)
        s = compute_summary(sig)
        text = render_summary_stats(s)
        assert "Total events" in text
        assert "Time span" in text
        assert "Mean rate" in text
        assert "Peak rate" in text


# ---------------------------------------------------------------------------
# SpectralPeak describe with new fields
# ---------------------------------------------------------------------------

class TestPeakDescribe:
    def test_describe_with_ci(self):
        p = SpectralPeak(
            frequency=1/600, period=600.0, magnitude=10.0, phase=0.0,
            rank=1, confidence_low=580.0, confidence_high=620.0,
        )
        desc = p.describe()
        assert "95% CI" in desc
        assert "580" in desc or "9.7m" in desc  # formatted as minutes

    def test_describe_with_harmonics(self):
        p = SpectralPeak(
            frequency=1/600, period=600.0, magnitude=10.0, phase=0.0,
            rank=1, harmonics=[300.0, 1200.0],
        )
        desc = p.describe()
        assert "harmonics" in desc

    def test_describe_minimal(self):
        p = SpectralPeak(
            frequency=1/600, period=600.0, magnitude=10.0, phase=0.0,
            rank=1,
        )
        desc = p.describe()
        assert "period=" in desc
        assert "magnitude=" in desc


# ---------------------------------------------------------------------------
# CLI flags (integration-style)
# ---------------------------------------------------------------------------

class TestCLINewFlags:
    def test_analyze_with_window_flag(self):
        """CLI should accept --window flag."""
        from spectral.cli import main
        import io, sys
        lines = _generate_test_log()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("\n".join(lines))
            f.flush()
            try:
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    main(["analyze", f.name, "--window", "blackman"])
                except SystemExit:
                    pass
                sys.stderr = old_stderr
            finally:
                os.unlink(f.name)

    def test_analyze_with_export(self):
        """CLI --export should create a JSON file."""
        from spectral.cli import main
        import io, sys
        lines = _generate_test_log()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("\n".join(lines))
            f.flush()
            export_path = f.name + ".json"
            try:
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                try:
                    main(["analyze", f.name, "--export", export_path, "--window", "rectangular"])
                except SystemExit:
                    pass
                sys.stderr = old_stderr
                assert os.path.exists(export_path), "Export file was not created"
                with open(export_path) as ef:
                    data = json.load(ef)
                assert isinstance(data, list)
                if data:
                    assert "signal_name" in data[0]
                    assert "anomalies" in data[0]
            finally:
                os.unlink(f.name)
                if os.path.exists(export_path):
                    os.unlink(export_path)

    def test_analyze_with_summary(self):
        """CLI --summary should not crash."""
        from spectral.cli import main
        import io, sys
        lines = _generate_test_log()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("\n".join(lines))
            f.flush()
            try:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                try:
                    main(["analyze", f.name, "--summary"])
                except SystemExit:
                    pass
                output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                assert "Summary Statistics" in output
            finally:
                os.unlink(f.name)


def _generate_test_log():
    """Generate simple test log lines."""
    from datetime import datetime, timedelta
    base = datetime(2024, 1, 1, 0, 0, 0)
    lines = []
    for i in range(200):
        ts = base + timedelta(seconds=i * 10)
        level = "INFO" if i % 5 != 0 else "ERROR"
        lines.append(f"{ts.isoformat()} [{level}] Event {i}")
    return lines
