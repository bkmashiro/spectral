"""
Convert log events into discrete time-series signals suitable for FFT.

Each event category (level, source, or custom grouping) becomes a separate
channel. The time axis is divided into uniform buckets and events are
counted per bucket.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from .parser import LogEvent


@dataclass
class Signal:
    """A discrete time-series: event counts per time bucket."""
    name: str
    bucket_size: float          # seconds per bucket
    start_time: float           # epoch of first bucket
    values: list[float]         # count (or rate) per bucket

    @property
    def duration(self) -> float:
        return len(self.values) * self.bucket_size

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

    @property
    def num_samples(self) -> int:
        return len(self.values)

    @property
    def sample_rate(self) -> float:
        """Samples per second."""
        return 1.0 / self.bucket_size if self.bucket_size > 0 else 0.0

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def total(self) -> float:
        return sum(self.values)


GroupKeyFn = Callable[[LogEvent], str]


def group_by_level(event: LogEvent) -> str:
    """Group events by log level."""
    return event.level.upper() if event.level else "UNKNOWN"


def group_by_source(event: LogEvent) -> str:
    """Group events by source/module."""
    return event.source if event.source else "UNKNOWN"


def group_all(_event: LogEvent) -> str:
    """Put all events in one group."""
    return "ALL"


def _auto_bucket_size(events: list[LogEvent]) -> float:
    """Choose a reasonable bucket size based on time span and event density."""
    if len(events) < 2:
        return 1.0
    times = sorted(e.timestamp for e in events)
    span = times[-1] - times[0]
    if span <= 0:
        return 1.0

    # Aim for ~500-2000 buckets for good spectral resolution
    target_buckets = 1024
    raw = span / target_buckets

    # Snap to nice intervals: 1s, 5s, 10s, 30s, 60s, 300s, 600s, 3600s
    nice = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 86400]
    for n in nice:
        if n >= raw:
            return float(n)
    return raw


def events_to_signals(
    events: list[LogEvent],
    group_fn: GroupKeyFn = group_all,
    bucket_size: float | None = None,
    normalize: bool = False,
) -> list[Signal]:
    """
    Convert a list of log events into time-bucketed signals.

    Args:
        events: Parsed log events (need not be sorted).
        group_fn: Function mapping event -> group name.
        bucket_size: Seconds per bucket. Auto-detected if None.
        normalize: If True, divide counts by bucket_size to get rate/sec.

    Returns:
        List of Signal objects, one per group.
    """
    if not events:
        return []

    if bucket_size is None:
        bucket_size = _auto_bucket_size(events)

    # Find time range
    times = [e.timestamp for e in events]
    t_min = min(times)
    t_max = max(times)
    n_buckets = max(1, math.ceil((t_max - t_min) / bucket_size) + 1)

    # Group events
    groups: dict[str, list[float]] = {}
    for event in events:
        key = group_fn(event)
        if key not in groups:
            groups[key] = [0.0] * n_buckets
        idx = min(int((event.timestamp - t_min) / bucket_size), n_buckets - 1)
        groups[key][idx] += 1.0

    # Build signals
    signals = []
    for name, values in sorted(groups.items()):
        if normalize:
            values = [v / bucket_size for v in values]
        signals.append(Signal(
            name=name,
            bucket_size=bucket_size,
            start_time=t_min,
            values=values,
        ))

    return signals


def pad_to_power_of_2(signal: Signal) -> Signal:
    """Zero-pad a signal so its length is a power of 2 (improves FFT performance)."""
    n = len(signal.values)
    if n == 0:
        return signal
    target = 1
    while target < n:
        target *= 2
    if target == n:
        return signal
    padded = signal.values + [0.0] * (target - n)
    return Signal(
        name=signal.name,
        bucket_size=signal.bucket_size,
        start_time=signal.start_time,
        values=padded,
    )
