"""
Spectral analysis: FFT, peak detection, periodicity finding, anomaly detection.

Uses only the Python standard library (cmath) for FFT computation --
no numpy required.

Supports window functions (hamming, hanning, blackman, rectangular),
harmonic detection, confidence intervals via bootstrap resampling,
and z-score based anomaly marking.
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .signal import Signal, pad_to_power_of_2


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def _window_rectangular(n: int) -> list[float]:
    """Rectangular (no) window."""
    return [1.0] * n


def _window_hamming(n: int) -> list[float]:
    """Hamming window."""
    if n <= 1:
        return [1.0] * n
    return [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def _window_hanning(n: int) -> list[float]:
    """Hanning (Hann) window."""
    if n <= 1:
        return [1.0] * n
    return [0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]


def _window_blackman(n: int) -> list[float]:
    """Blackman window."""
    if n <= 1:
        return [1.0] * n
    return [
        0.42 - 0.5 * math.cos(2 * math.pi * i / (n - 1))
        + 0.08 * math.cos(4 * math.pi * i / (n - 1))
        for i in range(n)
    ]


WINDOW_FUNCTIONS = {
    "rectangular": _window_rectangular,
    "hamming": _window_hamming,
    "hanning": _window_hanning,
    "blackman": _window_blackman,
}


def apply_window(values: list[float], window_name: str = "hamming") -> list[float]:
    """Apply a window function to a list of values."""
    func = WINDOW_FUNCTIONS.get(window_name, _window_hamming)
    window = func(len(values))
    return [v * w for v, w in zip(values, window)]


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
    confidence_low: Optional[float] = None   # 95% CI lower bound on period
    confidence_high: Optional[float] = None  # 95% CI upper bound on period
    harmonics: list[float] = field(default_factory=list)  # detected harmonic periods

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

        parts = [f"period={self.period_human}{hint}, magnitude={self.magnitude:.2f}"]

        if self.confidence_low is not None and self.confidence_high is not None:
            lo = _format_period(self.confidence_low)
            hi = _format_period(self.confidence_high)
            parts.append(f"95% CI: {lo}–{hi}")

        if self.harmonics:
            harm_strs = [_format_period(h) for h in self.harmonics[:3]]
            parts.append(f"harmonics: {', '.join(harm_strs)}")

        return ", ".join(parts)


def _format_period(s: float) -> str:
    """Format a period in seconds as human-readable."""
    if s < 60:
        return f"{s:.1f}s"
    elif s < 3600:
        return f"{s / 60:.1f}m"
    elif s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"


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
    window_function: str = "hamming"  # which window was used

    @property
    def dominant_period(self) -> float | None:
        """Period of the strongest peak, if any."""
        return self.peaks[0].period if self.peaks else None

    def summary(self) -> str:
        lines = [f"=== Spectral Analysis: {self.signal_name} ==="]
        lines.append(f"  Samples: {self.n_samples}, Sample rate: {self.sample_rate:.4f} Hz")
        lines.append(f"  Window function: {self.window_function}")
        lines.append(f"  DC (mean rate): {self.dc_component:.2f}")
        lines.append(f"  Spectral energy: {self.spectral_energy:.2f}")
        if self.peaks:
            lines.append(f"  Detected periodicities ({len(self.peaks)}):")
            for p in self.peaks:
                lines.append(f"    #{p.rank}: {p.describe()}")
        else:
            lines.append("  No significant periodicities detected.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export analysis results as a dictionary suitable for JSON serialization."""
        return {
            "signal_name": self.signal_name,
            "sample_rate": self.sample_rate,
            "n_samples": self.n_samples,
            "window_function": self.window_function,
            "dc_component": self.dc_component,
            "spectral_energy": self.spectral_energy,
            "peaks": [
                {
                    "rank": p.rank,
                    "frequency": p.frequency,
                    "period": p.period,
                    "magnitude": p.magnitude,
                    "phase": p.phase,
                    "confidence_low": p.confidence_low,
                    "confidence_high": p.confidence_high,
                    "harmonics": p.harmonics,
                }
                for p in self.peaks
            ],
            "frequencies": self.frequencies,
            "magnitudes": self.magnitudes,
        }


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
    z_score: float = 0.0  # z-score relative to mean energy

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
# Harmonic detection
# ---------------------------------------------------------------------------

def _find_harmonics(
    peak_period: float,
    all_peaks: list[SpectralPeak],
    tolerance: float = 0.1,
) -> list[float]:
    """Check if harmonics (P/2, P/3, 2P, 3P) of a period exist in peaks."""
    harmonic_ratios = [0.5, 1.0 / 3, 2.0, 3.0]
    found = []
    for ratio in harmonic_ratios:
        expected = peak_period * ratio
        for p in all_peaks:
            if abs(p.period - expected) / max(expected, 0.001) < tolerance:
                found.append(p.period)
                break
    return found


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_period_ci(
    values: list[float],
    sample_rate: float,
    target_period: float,
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    window_name: str = "hamming",
) -> tuple[float, float]:
    """Estimate confidence interval for a detected period via bootstrap resampling.

    Resamples the signal (with replacement of blocks) and re-runs FFT to
    find where the peak lands each time.
    """
    n = len(values)
    if n < 16:
        return target_period, target_period

    block_size = max(8, n // 16)
    periods: list[float] = []
    rng = random.Random(42)  # deterministic for reproducibility

    for _ in range(n_bootstrap):
        # Block bootstrap
        resampled = []
        while len(resampled) < n:
            start = rng.randint(0, n - block_size)
            resampled.extend(values[start:start + block_size])
        resampled = resampled[:n]

        # Apply window and FFT
        windowed = apply_window(resampled, window_name)
        mean_val = sum(windowed) / len(windowed)
        centered = [v - mean_val for v in windowed]

        # Pad to power of 2
        target_n = 1
        while target_n < len(centered):
            target_n *= 2
        padded = centered + [0.0] * (target_n - len(centered))

        spec = _fft_recursive([complex(v) for v in padded])
        half = len(spec) // 2
        freq_res = sample_rate / len(spec)

        if half <= 1:
            continue

        mags = [abs(spec[i]) / half for i in range(1, half)]

        # Find peak near target period
        target_freq = 1.0 / target_period if target_period > 0 else 0
        best_mag = 0
        best_period = target_period
        for i, mag in enumerate(mags):
            freq = (i + 1) * freq_res
            if freq > 0 and mag > best_mag:
                period = 1.0 / freq
                # Only consider peaks within 50% of target
                if 0.5 * target_period <= period <= 2.0 * target_period:
                    best_mag = mag
                    best_period = period

        periods.append(best_period)

    if not periods:
        return target_period, target_period

    periods.sort()
    alpha = (1 - confidence) / 2
    lo_idx = max(0, int(alpha * len(periods)))
    hi_idx = min(len(periods) - 1, int((1 - alpha) * len(periods)))
    return periods[lo_idx], periods[hi_idx]


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

@dataclass
class SignalSummary:
    """Summary statistics for a signal."""
    total_events: int
    time_span_seconds: float
    mean_rate: float       # events per second
    peak_rate: float       # max bucket value
    quietest_rate: float   # min bucket value
    bucket_size: float


def compute_summary(signal: Signal) -> SignalSummary:
    """Compute summary statistics for a signal."""
    total = signal.total
    span = signal.duration
    mean_rate = total / span if span > 0 else 0
    peak = max(signal.values) if signal.values else 0
    quietest = min(signal.values) if signal.values else 0
    return SignalSummary(
        total_events=int(total),
        time_span_seconds=span,
        mean_rate=mean_rate / signal.bucket_size if signal.bucket_size > 0 else 0,
        peak_rate=peak / signal.bucket_size if signal.bucket_size > 0 else 0,
        quietest_rate=quietest / signal.bucket_size if signal.bucket_size > 0 else 0,
        bucket_size=signal.bucket_size,
    )


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def analyze_signal(
    signal: Signal,
    max_peaks: int = 10,
    min_prominence: float = 0.1,
    window_name: str = "hamming",
    compute_confidence: bool = False,
    detect_harmonics: bool = True,
) -> SpectralResult:
    """
    Perform FFT analysis on a signal and extract spectral peaks.

    Args:
        signal: Input time-series signal.
        max_peaks: Maximum number of peaks to return.
        min_prominence: Minimum relative prominence (fraction of max magnitude)
                       for a peak to be reported.
        window_name: Window function to apply before FFT.
        compute_confidence: Whether to compute bootstrap confidence intervals.
        detect_harmonics: Whether to detect harmonics of each peak.
    """
    padded = pad_to_power_of_2(signal)
    values = padded.values
    n = len(values)

    if n == 0:
        return SpectralResult(
            signal_name=signal.name, sample_rate=0, n_samples=0,
            frequencies=[], magnitudes=[], phases=[], peaks=[],
            dc_component=0, spectral_energy=0, window_function=window_name,
        )

    # Remove DC offset (mean) before FFT for better peak detection
    mean_val = sum(values) / n
    centered = [v - mean_val for v in values]

    # Apply window function
    windowed = apply_window(centered, window_name)

    spectrum = fft(windowed)

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

        # Sort by magnitude descending, take top N
        raw_peaks.sort(key=lambda p: p.magnitude, reverse=True)
        top_peaks = raw_peaks[:max_peaks]

        # Detect harmonics
        peaks = []
        for rank, p in enumerate(top_peaks, 1):
            harmonics = []
            if detect_harmonics:
                harmonics = _find_harmonics(p.period, top_peaks)

            ci_low = None
            ci_high = None
            if compute_confidence and len(signal.values) >= 32:
                ci_low, ci_high = _bootstrap_period_ci(
                    signal.values, sample_rate, p.period,
                    window_name=window_name,
                )

            peaks.append(SpectralPeak(
                frequency=p.frequency,
                period=p.period,
                magnitude=p.magnitude,
                phase=p.phase,
                rank=rank,
                confidence_low=ci_low,
                confidence_high=ci_high,
                harmonics=harmonics,
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
        window_function=window_name,
    )


def detect_anomalies(
    signal: Signal,
    window_buckets: int = 64,
    step_buckets: int = 16,
    threshold: float = 2.5,
    z_score_threshold: float = 3.0,
) -> list[Anomaly]:
    """
    Detect anomalies by comparing spectral energy in sliding windows
    against a baseline. Also marks events more than z_score_threshold
    standard deviations from the mean frequency.

    Args:
        signal: Input signal.
        window_buckets: Number of buckets per analysis window.
        step_buckets: Step size between windows.
        threshold: Energy ratio above which a window is anomalous.
        z_score_threshold: Z-score threshold for anomaly marking.

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

    # Compute mean and std for z-score
    mean_energy = sum(e for _, e in window_energies) / len(window_energies)
    std_energy = (
        sum((e - mean_energy) ** 2 for _, e in window_energies) / len(window_energies)
    ) ** 0.5
    if std_energy == 0:
        std_energy = 1.0

    anomalies = []
    for start_bucket, energy in window_energies:
        ratio = energy / baseline
        z_score = (energy - mean_energy) / std_energy

        if ratio >= threshold or abs(z_score) >= z_score_threshold:
            t_start = signal.start_time + start_bucket * signal.bucket_size
            t_end = t_start + window_buckets * signal.bucket_size
            anomalies.append(Anomaly(
                window_start=t_start,
                window_end=t_end,
                energy_ratio=ratio,
                description=(
                    f"Spectral energy {ratio:.1f}x baseline "
                    f"(z={z_score:.1f}) in "
                    f"{window_buckets * signal.bucket_size:.0f}s window"
                ),
                z_score=z_score,
            ))

    return anomalies
