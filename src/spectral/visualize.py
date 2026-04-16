"""
Terminal-based visualization using Unicode block characters.

Renders:
- Spectrograms (frequency vs time, intensity as brightness)
- Frequency spectra (magnitude vs frequency bar chart)
- Time-series waveforms
- Anomaly timeline
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .analysis import Anomaly, SpectralResult, SignalSummary
from .signal import Signal

# Unicode block characters for vertical bar charts (⅛ increments)
_VBLOCKS = " ▁▂▃▄▅▆▇█"

# Unicode block characters for heat intensity
_HEAT = " ░▒▓█"


def _bar_char(value: float, max_val: float) -> str:
    """Map a value [0, max_val] to a vertical block character."""
    if max_val <= 0 or value <= 0:
        return _VBLOCKS[0]
    frac = min(value / max_val, 1.0)
    idx = int(frac * (len(_VBLOCKS) - 1))
    return _VBLOCKS[idx]


def _heat_char(value: float, max_val: float) -> str:
    """Map a value to a heat/intensity character."""
    if max_val <= 0 or value <= 0:
        return _HEAT[0]
    frac = min(value / max_val, 1.0)
    idx = int(frac * (len(_HEAT) - 1))
    return _HEAT[idx]


def _format_time(epoch: float) -> str:
    """Format epoch as compact time string."""
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.strftime("%H:%M:%S")


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


# ---------------------------------------------------------------------------
# Spectrum plot (frequency domain)
# ---------------------------------------------------------------------------

def render_spectrum(
    result: SpectralResult,
    width: int = 72,
    height: int = 20,
    max_freq: float | None = None,
) -> str:
    """
    Render a frequency spectrum as a horizontal bar chart.

    Each row is a frequency bin; bar length represents magnitude.
    Peaks are highlighted with markers.
    """
    if not result.magnitudes:
        return "(no data)"

    mags = result.magnitudes[1:]  # skip DC
    freqs = result.frequencies[1:]
    if not mags:
        return "(no data)"

    if max_freq is not None:
        cutoff = next((i for i, f in enumerate(freqs) if f > max_freq), len(freqs))
        mags = mags[:cutoff]
        freqs = freqs[:cutoff]

    # Downsample to fit height
    n = len(mags)
    if n > height:
        step = n / height
        ds_mags = []
        ds_freqs = []
        for i in range(height):
            start = int(i * step)
            end = int((i + 1) * step)
            chunk = mags[start:end]
            ds_mags.append(max(chunk) if chunk else 0)
            ds_freqs.append(freqs[start])
        mags = ds_mags
        freqs = ds_freqs

    max_mag = max(mags) if mags else 1.0
    peak_freqs = {p.frequency for p in result.peaks}
    label_w = 10

    lines = []
    lines.append(f"{'Freq':>{label_w}} │ Magnitude")
    lines.append(f"{'─' * label_w}─┼─{'─' * width}")

    for freq, mag in zip(freqs, mags):
        bar_len = int(mag / max_mag * width) if max_mag > 0 else 0
        period = 1.0 / freq if freq > 0 else float("inf")

        # Check if this bin is near a peak
        is_peak = any(abs(freq - pf) / max(freq, 0.001) < 0.1 for pf in peak_freqs)
        marker = "◆" if is_peak else "█"

        label = _format_duration(period)
        bar = marker * bar_len
        lines.append(f"{label:>{label_w}} │ {bar}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Time series waveform
# ---------------------------------------------------------------------------

def render_waveform(
    signal: Signal,
    width: int = 72,
    height: int = 12,
) -> str:
    """Render a signal as a time-series waveform using block characters."""
    if not signal.values:
        return "(no data)"

    values = signal.values
    n = len(values)

    # Downsample to width
    if n > width:
        step = n / width
        ds = []
        for i in range(width):
            start = int(i * step)
            end = int((i + 1) * step)
            chunk = values[start:end]
            ds.append(max(chunk) if chunk else 0)
        values = ds

    max_val = max(values) if values else 1.0

    lines = []
    lines.append(f"  Signal: {signal.name}  "
                 f"(bucket={_format_duration(signal.bucket_size)}, "
                 f"span={_format_duration(signal.duration)})")

    # Build vertical bars
    bar_line = " ".join(_bar_char(v, max_val) for v in values[:width])
    lines.append(f"  {bar_line}")

    # Time axis labels
    t_start = _format_time(signal.start_time)
    t_end = _format_time(signal.end_time)
    axis = f"  {t_start}{' ' * max(0, len(bar_line) - len(t_start) - len(t_end))}{t_end}"
    lines.append(axis)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spectrogram (time x frequency heat map)
# ---------------------------------------------------------------------------

def render_spectrogram(
    signal: Signal,
    window_size: int = 64,
    step: int = 8,
    width: int = 72,
    height: int = 24,
    max_freq_bins: int | None = None,
) -> str:
    """
    Render a spectrogram: time on x-axis, frequency on y-axis, intensity as
    Unicode block shading.
    """
    from .analysis import fft

    values = signal.values
    n = len(values)

    if n < window_size:
        return "(signal too short for spectrogram)"

    # Compute STFT
    columns: list[list[float]] = []
    for start in range(0, n - window_size + 1, step):
        window = values[start:start + window_size]
        mean_val = sum(window) / len(window)
        centered = [v - mean_val for v in window]
        spec = fft(centered)
        half = len(spec) // 2
        col = [abs(spec[i]) / half for i in range(1, half)]
        columns.append(col)

    if not columns:
        return "(no spectrogram data)"

    freq_bins = len(columns[0])
    time_bins = len(columns)

    # Limit frequency bins
    if max_freq_bins and freq_bins > max_freq_bins:
        freq_bins = max_freq_bins
        columns = [col[:freq_bins] for col in columns]

    # Downsample to fit display
    if time_bins > width:
        t_step = time_bins / width
        new_cols = []
        for i in range(width):
            s = int(i * t_step)
            e = int((i + 1) * t_step)
            merged = [max(columns[j][k] for j in range(s, min(e, time_bins)))
                      for k in range(freq_bins)]
            new_cols.append(merged)
        columns = new_cols
        time_bins = width

    if freq_bins > height:
        f_step = freq_bins / height
        new_cols = []
        for col in columns:
            merged = []
            for i in range(height):
                s = int(i * f_step)
                e = int((i + 1) * f_step)
                merged.append(max(col[s:e]) if col[s:e] else 0)
            new_cols.append(merged)
        columns = new_cols
        freq_bins = height

    # Find global max
    global_max = max(max(col) for col in columns) if columns else 1.0

    lines = []
    lines.append(f"  Spectrogram: {signal.name}")
    lines.append(f"  ↑ freq")

    # Render rows (high freq at top)
    for fi in range(freq_bins - 1, -1, -1):
        row = "".join(_heat_char(col[fi], global_max) for col in columns)
        lines.append(f"  │{row}│")

    lines.append(f"  └{'─' * time_bins}┘")
    t_start = _format_time(signal.start_time)
    t_end = _format_time(signal.end_time)
    lines.append(f"  {t_start}{' ' * max(0, time_bins - len(t_start) - len(t_end) + 2)}{t_end}")
    lines.append(f"  → time")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anomaly timeline
# ---------------------------------------------------------------------------

def render_anomaly_timeline(
    anomalies: list[Anomaly],
    signal: Signal,
    width: int = 72,
) -> str:
    """Render anomalies on a timeline."""
    if not anomalies:
        return "  No anomalies detected."

    lines = []
    lines.append(f"  Anomaly Timeline ({len(anomalies)} detected)")

    span = signal.duration
    if span <= 0:
        return "  (zero-length signal)"

    # Build timeline strip
    strip = [" "] * width
    for a in anomalies:
        rel_start = (a.window_start - signal.start_time) / span
        rel_end = (a.window_end - signal.start_time) / span
        i_start = max(0, min(int(rel_start * width), width - 1))
        i_end = max(0, min(int(rel_end * width), width))
        char = "█" if a.severity in ("CRITICAL", "HIGH") else "▓" if a.severity == "MEDIUM" else "░"
        for i in range(i_start, i_end):
            strip[i] = char

    lines.append(f"  ┌{'─' * width}┐")
    lines.append(f"  │{''.join(strip)}│")
    lines.append(f"  └{'─' * width}┘")
    t_start = _format_time(signal.start_time)
    t_end = _format_time(signal.end_time)
    lines.append(f"  {t_start}{' ' * max(0, width - len(t_start) - len(t_end) + 2)}{t_end}")

    lines.append("")
    for a in anomalies:
        t = _format_time(a.window_start)
        lines.append(f"  [{a.severity:>8}] {t} - {a.description}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def render_summary_stats(summary: SignalSummary) -> str:
    """Render summary statistics for a signal."""
    lines = []
    lines.append("  Summary Statistics:")
    lines.append(f"    Total events:    {summary.total_events:,}")
    lines.append(f"    Time span:       {_format_duration(summary.time_span_seconds)}")
    lines.append(f"    Bucket size:     {_format_duration(summary.bucket_size)}")
    lines.append(f"    Mean rate:       {summary.mean_rate:.2f}/s")
    lines.append(f"    Peak rate:       {summary.peak_rate:.2f}/s")
    lines.append(f"    Quietest rate:   {summary.quietest_rate:.2f}/s")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def render_full_report(
    signal: Signal,
    result: SpectralResult,
    anomalies: list[Anomaly],
    width: int = 72,
) -> str:
    """Render a complete analysis report."""
    sections = []
    sections.append("=" * width)
    sections.append(f"  SPECTRAL ANALYSIS REPORT: {signal.name}")
    sections.append("=" * width)
    sections.append("")
    sections.append(render_waveform(signal, width=width))
    sections.append("")
    sections.append(result.summary())
    sections.append("")
    sections.append(render_spectrum(result, width=max(40, width - 14)))
    sections.append("")
    sections.append(render_anomaly_timeline(anomalies, signal, width=width))
    return "\n".join(sections)
