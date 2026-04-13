"""
Spectral analysis: FFT, peak detection, periodicity finding, anomaly detection.

Uses only the Python standard library (cmath) for FFT computation --
no numpy required.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

from .signal import Signal, pad_to_power_of_2


# ---------------------------------------------------------------------------
# Pure-Python FFT (Cooley-Tukey radix-2 DIT)
# ---------------------------------------------------------------------------

def _fft_recursive(x: list[complex]) -> list[complex]:
    """Radix-2 decimation-in-time FFT. Input length must be a power of 2."""
    n = len(x)
    if n <= 1:
        return x
    even = _fft_recursive(x[0::2])
    odd = _fft_recursive(x[1::2])
    T = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + \
           [even[k] - T[k] for k in range(n // 2)]


def fft(values: list[float]) -> list[complex]:
    """Compute FFT of real-valued input. Pads to power of 2 if needed."""
    n = len(values)
    # Pad to power of 2
    target = 1
    while target < n:
        target *= 2
    padded = values + [0.0] * (target - n)
    return _fft_recursive([complex(v) for v in padded])


# ---------------------------------------------------------------------------
# Spectral analysis results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectralPeak:
    """A peak in the frequency spectrum."""
    frequency: float      # Hz (cycles per second)
    period: float         # seconds
    magnitude: float      # amplitude
    phase: float          # radians
    rank: int = 0         # 1 = strongest, 2 = second, etc.

    @property
    def period_human(self) -> str:
        """Human-readable period string."""
        s = self.period
        if s < 60:
            return f"{s:.1f}s"
        elif s < 3600:
            return f"{s / 60:.1f}m"
        elif s < 86400:
            return f"{s / 3600:.1f}h"
        else:
            return f"{s / 86400:.1f}d"

    def describe(self) -> str:
        """Describe this peak with possible interpretation."""
        p = self.period
        hints = []
        # Check for common intervals
        known = [
            (60, "every minute"),
            (300, "every 5 minutes"),
            (600, "every 10 minutes"),
            (900, "every 15 minutes"),
            (1800, "every 30 minutes"),
            (3600, "hourly"),
            (7200, "every 2 hours"),
            (14400, "every 4 hours"),
            (21600, "every 6 hours"),
            (43200, "every 12 hours"),
            (86400, "daily"),
            (604800, "weekly"),
        ]
        for interval, label in known:
            if abs(p - interval) / interval < 0.05:  # within 5%
                hints.append(label)
                break
        hint = f" ({hints[0]})" if hints else ""
        return f"period={self.period_human}{hint}, magnitude={self.magnitude:.2f}"


@dataclass
class SpectralResult:
    """Complete spectral analysis of a signal."""
    signal_name: str
    sample_rate: float
    n_samples: int
    frequencies: list[float]
    magnitudes: list[float]
    phases: list[float]
    peaks: list[SpectralPeak]
    dc_component: float     # mean level (magnitude at freq 0)
    spectral_energy: float  # total energy in spectrum

    @property
    def dominant_period(self) -> float | None:
        """Period of the strongest peak, if any."""
        return self.peaks[0].period if self.peaks else None

    def summary(self) -> str:
        lines = [f"=== Spectral Analysis: {self.signal_name} ==="]
        lines.append(f"  Samples: {self.n_samples}, Sample rate: {self.sample_rate:.4f} Hz")
        lines.append(f"  DC (mean rate): {self.dc_component:.2f}")
        lines.append(f"  Spectral energy: {self.spectral_energy:.2f}")
        if self.peaks:
            lines.append(f"  Detected periodicities ({len(self.peaks)}):")
            for p in self.peaks:
                lines.append(f"    #{p.rank}: {p.describe()}")
        else:
            lines.append("  No significant periodicities detected.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anomaly detection via sliding-window spectral comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Anomaly:
    """A detected spectral anomaly in a time window."""
    window_start: float   # epoch seconds
    window_end: float
    energy_ratio: float   # ratio of this window's energy to baseline
    description: str

    @property
    def severity(self) -> str:
        if self.energy_ratio > 5.0:
            return "CRITICAL"
        elif self.energy_ratio > 3.0:
            return "HIGH"
        elif self.energy_ratio > 2.0:
            return "MEDIUM"
        return "LOW"


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def analyze_signal(
    signal: Signal,
    max_peaks: int = 10,
    min_prominence: float = 0.1,
) -> SpectralResult:
    """
    Perform FFT analysis on a signal and extract spectral peaks.

    Args:
        signal: Input time-series signal.
        max_peaks: Maximum number of peaks to return.
        min_prominence: Minimum relative prominence (fraction of max magnitude)
                       for a peak to be reported.
    """
    padded = pad_to_power_of_2(signal)
    values = padded.values
    n = len(values)

    if n == 0:
        return SpectralResult(
            signal_name=signal.name, sample_rate=0, n_samples=0,
            frequencies=[], magnitudes=[], phases=[], peaks=[],
            dc_component=0, spectral_energy=0,
        )

    # Remove DC offset (mean) before FFT for better peak detection
    mean_val = sum(values) / n
    centered = [v - mean_val for v in values]

    spectrum = fft(centered)

    # Only use first half (positive frequencies) for real input
    half_n = n // 2
    sample_rate = signal.sample_rate
    freq_resolution = sample_rate / n

    frequencies = [i * freq_resolution for i in range(half_n)]
    magnitudes = [abs(spectrum[i]) / half_n for i in range(half_n)]
    phases = [cmath.phase(spectrum[i]) for i in range(half_n)]

    dc_component = abs(spectrum[0]) / n  # True DC from un-centered would be mean

    spectral_energy = sum(m * m for m in magnitudes)

    # Peak detection: find local maxima above threshold
    if not magnitudes or max(magnitudes) == 0:
        peaks = []
    else:
        max_mag = max(magnitudes[1:]) if len(magnitudes) > 1 else 0  # skip DC bin
        threshold = max_mag * min_prominence

        raw_peaks = []
        for i in range(1, half_n - 1):  # skip DC (i=0) and Nyquist
            if magnitudes[i] > threshold:
                if magnitudes[i] >= magnitudes[i - 1] and magnitudes[i] >= magnitudes[i + 1]:
                    freq = frequencies[i]
                    period = 1.0 / freq if freq > 0 else float("inf")
                    raw_peaks.append(SpectralPeak(
                        frequency=freq,
                        period=period,
                        magnitude=magnitudes[i],
                        phase=phases[i],
                    ))

        # Sort by magnitude descending, take top N, assign ranks
        raw_peaks.sort(key=lambda p: p.magnitude, reverse=True)
        peaks = []
        for rank, p in enumerate(raw_peaks[:max_peaks], 1):
            peaks.append(SpectralPeak(
                frequency=p.frequency,
                period=p.period,
                magnitude=p.magnitude,
                phase=p.phase,
                rank=rank,
            ))

    return SpectralResult(
        signal_name=signal.name,
        sample_rate=sample_rate,
        n_samples=signal.num_samples,
        frequencies=frequencies,
        magnitudes=magnitudes,
        phases=phases,
        peaks=peaks,
        dc_component=mean_val,
        spectral_energy=spectral_energy,
    )


def detect_anomalies(
    signal: Signal,
    window_buckets: int = 64,
    step_buckets: int = 16,
    threshold: float = 2.5,
) -> list[Anomaly]:
    """
    Detect anomalies by comparing spectral energy in sliding windows
    against a baseline.

    Args:
        signal: Input signal.
        window_buckets: Number of buckets per analysis window.
        step_buckets: Step size between windows.
        threshold: Energy ratio above which a window is anomalous.

    Returns:
        List of detected anomalies.
    """
    values = signal.values
    n = len(values)
    if n < window_buckets * 2:
        return []

    # Compute spectral energy for each window
    window_energies: list[tuple[int, float]] = []
    for start in range(0, n - window_buckets + 1, step_buckets):
        window = values[start:start + window_buckets]
        mean_val = sum(window) / len(window)
        centered = [v - mean_val for v in window]
        spec = fft(centered)
        half = len(spec) // 2
        energy = sum(abs(spec[i]) ** 2 for i in range(1, half)) / (half * half)
        window_energies.append((start, energy))

    if not window_energies:
        return []

    # Baseline: median energy
    energies = sorted(e for _, e in window_energies)
    median_idx = len(energies) // 2
    baseline = energies[median_idx] if energies[median_idx] > 0 else 1.0

    anomalies = []
    for start_bucket, energy in window_energies:
        ratio = energy / baseline
        if ratio >= threshold:
            t_start = signal.start_time + start_bucket * signal.bucket_size
            t_end = t_start + window_buckets * signal.bucket_size
            anomalies.append(Anomaly(
                window_start=t_start,
                window_end=t_end,
                energy_ratio=ratio,
                description=(
                    f"Spectral energy {ratio:.1f}x baseline in "
                    f"{window_buckets * signal.bucket_size:.0f}s window"
                ),
            ))

    return anomalies
