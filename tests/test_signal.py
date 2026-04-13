"""Tests for signal construction."""

import math

from spectral.parser import LogEvent
from spectral.signal import (
    Signal,
    events_to_signals,
    group_all,
    group_by_level,
    group_by_source,
    pad_to_power_of_2,
)


def _make_events(times: list[float], level: str = "INFO", source: str = "test") -> list[LogEvent]:
    return [LogEvent(timestamp=t, level=level, source=source) for t in times]


class TestSignal:
    def test_properties(self):
        sig = Signal(name="test", bucket_size=10.0, start_time=1000.0, values=[1, 2, 3, 4])
        assert sig.duration == 40.0
        assert sig.end_time == 1040.0
        assert sig.num_samples == 4
        assert sig.sample_rate == 0.1
        assert sig.mean == 2.5
        assert sig.total == 10.0

    def test_empty_signal(self):
        sig = Signal(name="empty", bucket_size=1.0, start_time=0.0, values=[])
        assert sig.mean == 0.0
        assert sig.total == 0.0


class TestEventsToSignals:
    def test_basic_bucketing(self):
        # 10 events spread over 100 seconds, 10s buckets
        events = _make_events([float(i * 10) for i in range(10)])
        signals = events_to_signals(events, group_fn=group_all, bucket_size=10.0)
        assert len(signals) == 1
        sig = signals[0]
        assert sig.name == "ALL"
        assert sig.bucket_size == 10.0
        assert sum(sig.values) == 10  # all events accounted for

    def test_group_by_level(self):
        events = (
            _make_events([1.0, 2.0, 3.0], level="ERROR") +
            _make_events([1.5, 2.5], level="INFO")
        )
        signals = events_to_signals(events, group_fn=group_by_level, bucket_size=1.0)
        names = {s.name for s in signals}
        assert "ERROR" in names
        assert "INFO" in names

    def test_group_by_source(self):
        events = (
            _make_events([1.0, 2.0], source="api") +
            _make_events([1.0], source="db")
        )
        signals = events_to_signals(events, group_fn=group_by_source, bucket_size=1.0)
        names = {s.name for s in signals}
        assert "api" in names
        assert "db" in names

    def test_empty_events(self):
        signals = events_to_signals([])
        assert signals == []

    def test_single_event(self):
        events = _make_events([100.0])
        signals = events_to_signals(events, bucket_size=1.0)
        assert len(signals) == 1
        assert signals[0].total == 1.0

    def test_normalize(self):
        events = _make_events([float(i) for i in range(60)])
        signals = events_to_signals(events, bucket_size=10.0, normalize=True)
        sig = signals[0]
        # Normalized values should be events/second, not events/bucket
        assert all(v <= 1.0 for v in sig.values)  # ~1 event per second = 0.1/s


class TestPadToPowerOf2:
    def test_already_power_of_2(self):
        sig = Signal(name="t", bucket_size=1.0, start_time=0, values=[1.0] * 8)
        padded = pad_to_power_of_2(sig)
        assert len(padded.values) == 8

    def test_pads_correctly(self):
        sig = Signal(name="t", bucket_size=1.0, start_time=0, values=[1.0] * 5)
        padded = pad_to_power_of_2(sig)
        assert len(padded.values) == 8
        assert padded.values[5:] == [0.0, 0.0, 0.0]

    def test_empty(self):
        sig = Signal(name="t", bucket_size=1.0, start_time=0, values=[])
        padded = pad_to_power_of_2(sig)
        assert len(padded.values) == 0
