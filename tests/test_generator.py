"""Tests for the synthetic log generator."""

from spectral.generator import generate_log_lines, generate_json_log_lines, SEEDS
from spectral.parser import LogParser


class TestGenerateLogLines:
    def test_generates_lines(self):
        lines = generate_log_lines(duration_hours=0.5, seed=42)
        assert len(lines) > 0

    def test_deterministic(self):
        a = generate_log_lines(duration_hours=0.1, seed=42)
        b = generate_log_lines(duration_hours=0.1, seed=42)
        assert a == b

    def test_parseable(self):
        lines = generate_log_lines(duration_hours=0.5, seed=42)
        parser = LogParser()
        events = parser.parse_lines(lines)
        assert len(events) == len(lines)
        assert parser.detected_format == "iso"

    def test_has_multiple_levels(self):
        lines = generate_log_lines(duration_hours=1.0, seed=42)
        levels = set()
        for line in lines:
            parts = line.split(" ", 3)
            if len(parts) >= 2:
                levels.add(parts[1])
        assert len(levels) >= 3  # Should have INFO, WARN, ERROR at minimum

    def test_anomaly_injection(self):
        normal = generate_log_lines(duration_hours=1.0, seed=42, anomaly_at=None)
        with_anomaly = generate_log_lines(duration_hours=1.0, seed=42, anomaly_at=0.5)
        # Anomaly should produce more events
        assert len(with_anomaly) > len(normal)

    def test_custom_periods(self):
        lines = generate_log_lines(
            duration_hours=0.5,
            seed=42,
            periods=[(60.0, 0.5, "ERROR")],
        )
        assert len(lines) > 0

    def test_seeds_exist(self):
        assert len(SEEDS) == 16
        assert SEEDS[0] == 4825


class TestGenerateJSONLines:
    def test_generates_json(self):
        lines = generate_json_log_lines(duration_hours=0.5, seed=42)
        assert len(lines) > 0
        import json
        for line in lines[:5]:
            obj = json.loads(line)
            assert "timestamp" in obj
            assert "level" in obj
            assert "message" in obj

    def test_parseable(self):
        lines = generate_json_log_lines(duration_hours=0.5, seed=42)
        parser = LogParser()
        events = parser.parse_lines(lines)
        assert len(events) == len(lines)


class TestEndToEnd:
    def test_generate_parse_analyze(self):
        """Full pipeline: generate -> parse -> signal -> analyze."""
        from spectral.signal import events_to_signals, group_by_level
        from spectral.analysis import analyze_signal

        lines = generate_log_lines(duration_hours=2.0, seed=4825)
        parser = LogParser()
        events = parser.parse_lines(lines)
        signals = events_to_signals(events, group_fn=group_by_level)

        assert len(signals) > 0
        for sig in signals:
            result = analyze_signal(sig)
            assert result.signal_name == sig.name
            assert result.n_samples > 0
