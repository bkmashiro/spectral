"""Tests for visualization rendering."""

import math

from spectral.analysis import Anomaly, SpectralResult, analyze_signal
from spectral.signal import Signal
from spectral.visualize import (
    render_anomaly_timeline,
    render_full_report,
    render_spectrum,
    render_spectrogram,
    render_waveform,
)


def _make_test_signal(n: int = 256) -> Signal:
    values = [5.0 + 3.0 * math.sin(2 * math.pi * i / 64) for i in range(n)]
    return Signal(name="test", bucket_size=10.0, start_time=1000000.0, values=values)


class TestRenderWaveform:
    def test_basic(self):
        sig = _make_test_signal()
        output = render_waveform(sig)
        assert "test" in output
        assert "bucket=" in output

    def test_empty(self):
        sig = Signal(name="empty", bucket_size=1.0, start_time=0, values=[])
        output = render_waveform(sig)
        assert "no data" in output


class TestRenderSpectrum:
    def test_basic(self):
        sig = _make_test_signal()
        result = analyze_signal(sig)
        output = render_spectrum(result)
        assert "Freq" in output
        assert "Magnitude" in output

    def test_empty(self):
        result = SpectralResult(
            signal_name="empty", sample_rate=0, n_samples=0,
            frequencies=[], magnitudes=[], phases=[], peaks=[],
            dc_component=0, spectral_energy=0,
        )
        output = render_spectrum(result)
        assert "no data" in output

    def test_with_max_freq(self):
        sig = _make_test_signal()
        result = analyze_signal(sig)
        output = render_spectrum(result, max_freq=0.01)
        assert len(output) > 0


class TestRenderSpectrogram:
    def test_basic(self):
        sig = _make_test_signal(512)
        output = render_spectrogram(sig)
        assert "Spectrogram" in output

    def test_short_signal(self):
        sig = Signal(name="short", bucket_size=1.0, start_time=0, values=[1.0] * 10)
        output = render_spectrogram(sig)
        assert "too short" in output


class TestRenderAnomalyTimeline:
    def test_with_anomalies(self):
        sig = _make_test_signal()
        anomalies = [
            Anomaly(window_start=1001000.0, window_end=1001500.0,
                    energy_ratio=3.5, description="Spike detected"),
        ]
        output = render_anomaly_timeline(anomalies, sig)
        assert "1 detected" in output
        assert "HIGH" in output

    def test_no_anomalies(self):
        sig = _make_test_signal()
        output = render_anomaly_timeline([], sig)
        assert "No anomalies" in output


class TestRenderFullReport:
    def test_complete_report(self):
        sig = _make_test_signal()
        result = analyze_signal(sig)
        anomalies = [
            Anomaly(window_start=1001000.0, window_end=1001500.0,
                    energy_ratio=4.0, description="Test anomaly"),
        ]
        output = render_full_report(sig, result, anomalies)
        assert "SPECTRAL ANALYSIS REPORT" in output
        assert "test" in output
        assert "Freq" in output
