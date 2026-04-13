"""Tests for spectral analysis."""

import math

from spectral.analysis import analyze_signal, detect_anomalies, fft
from spectral.signal import Signal


class TestFFT:
    def test_constant_signal(self):
        """FFT of a constant should have DC component only."""
        result = fft([5.0] * 8)
        # DC component (index 0) should be N * value
        assert abs(result[0] - 40.0) < 1e-6
        # All other components should be ~0
        for i in range(1, len(result)):
            assert abs(result[i]) < 1e-6

    def test_pure_sine(self):
        """FFT of a pure sine should have a single peak."""
        n = 64
        freq = 4  # 4 cycles in the window
        signal = [math.sin(2 * math.pi * freq * i / n) for i in range(n)]
        result = fft(signal)
        magnitudes = [abs(c) for c in result]
        # Peak should be at index `freq` and `n - freq`
        assert magnitudes[freq] > 10  # strong peak
        # Other bins (excluding mirrors) should be small
        for i in range(1, n // 2):
            if i != freq:
                assert magnitudes[i] < 1.0

    def test_power_of_2_padding(self):
        """Non-power-of-2 input should be padded."""
        result = fft([1.0, 2.0, 3.0])
        assert len(result) == 4  # padded to 4

    def test_single_value(self):
        result = fft([7.0])
        assert len(result) == 1
        assert abs(result[0] - 7.0) < 1e-6

    def test_parseval_approx(self):
        """Energy in time domain ~ energy in frequency domain."""
        values = [math.sin(0.3 * i) + 0.5 * math.cos(0.7 * i) for i in range(64)]
        time_energy = sum(v * v for v in values)
        spec = fft(values)
        freq_energy = sum(abs(c) ** 2 for c in spec) / len(spec)
        assert abs(time_energy - freq_energy) / time_energy < 0.01


class TestAnalyzeSignal:
    def test_detects_known_period(self):
        """Should detect a sine wave's period."""
        bucket_size = 10.0  # 10 seconds per bucket
        period = 600.0      # 10-minute period
        n = 1024
        values = [
            10.0 + 5.0 * math.sin(2 * math.pi * i * bucket_size / period)
            for i in range(n)
        ]
        sig = Signal(name="test", bucket_size=bucket_size, start_time=0, values=values)
        result = analyze_signal(sig)

        assert len(result.peaks) > 0
        # The dominant peak should be near our injected period
        dominant = result.peaks[0]
        assert abs(dominant.period - period) / period < 0.05

    def test_multiple_periods(self):
        """Should detect multiple superimposed periods."""
        bucket_size = 5.0
        n = 2048
        period1 = 300.0   # 5 min
        period2 = 3600.0  # 1 hour
        values = [
            5.0 * math.sin(2 * math.pi * i * bucket_size / period1) +
            8.0 * math.sin(2 * math.pi * i * bucket_size / period2)
            for i in range(n)
        ]
        sig = Signal(name="multi", bucket_size=bucket_size, start_time=0, values=values)
        result = analyze_signal(sig, max_peaks=5)

        detected_periods = sorted([p.period for p in result.peaks])
        # Should find both periods within 10%
        found_300 = any(abs(p - period1) / period1 < 0.1 for p in detected_periods)
        found_3600 = any(abs(p - period2) / period2 < 0.1 for p in detected_periods)
        assert found_300, f"Did not find ~300s period in {detected_periods}"
        assert found_3600, f"Did not find ~3600s period in {detected_periods}"

    def test_empty_signal(self):
        sig = Signal(name="empty", bucket_size=1.0, start_time=0, values=[])
        result = analyze_signal(sig)
        assert result.peaks == []
        assert result.spectral_energy == 0

    def test_dc_component(self):
        """DC should reflect the mean."""
        values = [10.0] * 64
        sig = Signal(name="dc", bucket_size=1.0, start_time=0, values=values)
        result = analyze_signal(sig)
        assert abs(result.dc_component - 10.0) < 0.1

    def test_summary_string(self):
        values = [math.sin(i * 0.1) for i in range(256)]
        sig = Signal(name="test", bucket_size=1.0, start_time=0, values=values)
        result = analyze_signal(sig)
        summary = result.summary()
        assert "test" in summary
        assert "Samples" in summary


class TestSpectralPeak:
    def test_period_human_seconds(self):
        from spectral.analysis import SpectralPeak
        p = SpectralPeak(frequency=1.0, period=30.0, magnitude=1.0, phase=0.0, rank=1)
        assert p.period_human == "30.0s"

    def test_period_human_minutes(self):
        from spectral.analysis import SpectralPeak
        p = SpectralPeak(frequency=0.001, period=600.0, magnitude=1.0, phase=0.0, rank=1)
        assert p.period_human == "10.0m"

    def test_period_human_hours(self):
        from spectral.analysis import SpectralPeak
        p = SpectralPeak(frequency=0.0001, period=7200.0, magnitude=1.0, phase=0.0, rank=1)
        assert p.period_human == "2.0h"

    def test_describe_hourly(self):
        from spectral.analysis import SpectralPeak
        p = SpectralPeak(frequency=1/3600, period=3600.0, magnitude=5.0, phase=0.0, rank=1)
        desc = p.describe()
        assert "hourly" in desc

    def test_describe_daily(self):
        from spectral.analysis import SpectralPeak
        p = SpectralPeak(frequency=1/86400, period=86400.0, magnitude=5.0, phase=0.0, rank=1)
        desc = p.describe()
        assert "daily" in desc


class TestAnomalyDetection:
    def test_detects_spike(self):
        """A sudden spike in the signal should be flagged."""
        n = 512
        values = [1.0] * n
        # Inject a burst in the middle
        for i in range(200, 264):
            values[i] = 20.0

        sig = Signal(name="spike", bucket_size=10.0, start_time=0, values=values)
        anomalies = detect_anomalies(sig, window_buckets=64, step_buckets=16, threshold=2.0)
        assert len(anomalies) > 0
        # At least one anomaly should overlap the spike region
        spike_time = 200 * 10.0
        found = any(a.window_start <= spike_time <= a.window_end for a in anomalies)
        assert found

    def test_no_anomaly_in_uniform(self):
        """Uniform signal should have no anomalies."""
        values = [5.0] * 512
        sig = Signal(name="flat", bucket_size=1.0, start_time=0, values=values)
        anomalies = detect_anomalies(sig, threshold=2.5)
        assert len(anomalies) == 0

    def test_short_signal(self):
        """Too-short signals should return empty."""
        sig = Signal(name="short", bucket_size=1.0, start_time=0, values=[1.0] * 10)
        anomalies = detect_anomalies(sig)
        assert anomalies == []

    def test_anomaly_severity(self):
        from spectral.analysis import Anomaly
        a = Anomaly(window_start=0, window_end=100, energy_ratio=6.0, description="test")
        assert a.severity == "CRITICAL"
        a2 = Anomaly(window_start=0, window_end=100, energy_ratio=3.5, description="test")
        assert a2.severity == "HIGH"
        a3 = Anomaly(window_start=0, window_end=100, energy_ratio=2.2, description="test")
        assert a3.severity == "MEDIUM"
        a4 = Anomaly(window_start=0, window_end=100, energy_ratio=1.5, description="test")
        assert a4.severity == "LOW"
