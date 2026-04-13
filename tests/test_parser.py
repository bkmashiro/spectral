"""Tests for the log parser module."""

import json
import math
from datetime import datetime, timezone

from spectral.parser import (
    LogParser,
    parse_clf_line,
    parse_iso_line,
    parse_json_line,
    parse_syslog_line,
)


class TestISOParser:
    def test_basic_iso(self):
        line = "2024-01-15T08:30:00Z ERROR [auth] Token expired"
        event = parse_iso_line(line)
        assert event is not None
        assert event.level == "ERROR"
        assert event.source == "auth"
        assert event.message == "Token expired"

    def test_iso_with_millis(self):
        line = "2024-01-15T08:30:00.123Z INFO [api] Request OK"
        event = parse_iso_line(line)
        assert event is not None
        assert event.level == "INFO"

    def test_iso_with_offset(self):
        line = "2024-01-15T08:30:00+05:30 WARN [db] Slow query"
        event = parse_iso_line(line)
        assert event is not None
        assert event.level == "WARN"

    def test_iso_no_level(self):
        line = "2024-01-15T08:30:00Z Something happened"
        event = parse_iso_line(line)
        assert event is not None
        assert event.level == ""
        assert "Something happened" in event.message

    def test_iso_space_separator(self):
        line = "2024-01-15 08:30:00 ERROR [worker] Job failed"
        event = parse_iso_line(line)
        assert event is not None
        assert event.level == "ERROR"
        assert event.source == "worker"


class TestSyslogParser:
    def test_basic_syslog(self):
        line = "Apr 13 14:22:01 myhost sshd[1234]: Accepted publickey"
        event = parse_syslog_line(line)
        assert event is not None
        assert event.source == "sshd"
        assert "Accepted publickey" in event.message

    def test_syslog_no_pid(self):
        line = "Jan  5 03:00:00 server cron: Job started"
        event = parse_syslog_line(line)
        assert event is not None
        assert event.source == "cron"


class TestCLFParser:
    def test_basic_clf(self):
        line = '127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /index.html HTTP/1.0" 200 2326'
        event = parse_clf_line(line)
        assert event is not None
        assert event.level == "INFO"
        assert "GET" in event.message

    def test_clf_error_status(self):
        line = '10.0.0.1 - - [10/Oct/2000:13:55:36 -0700] "POST /api/data HTTP/1.1" 500 123'
        event = parse_clf_line(line)
        assert event is not None
        assert event.level == "ERROR"

    def test_clf_client_error(self):
        line = '10.0.0.1 - - [10/Oct/2000:13:55:36 -0700] "GET /missing HTTP/1.1" 404 0'
        event = parse_clf_line(line)
        assert event is not None
        assert event.level == "WARN"


class TestJSONParser:
    def test_basic_json(self):
        obj = {"timestamp": "2024-01-15T08:30:00Z", "level": "error", "message": "fail"}
        event = parse_json_line(json.dumps(obj))
        assert event is not None
        assert event.level == "ERROR"
        assert event.message == "fail"

    def test_json_numeric_ts(self):
        obj = {"timestamp": 1705305000, "level": "info", "msg": "ok"}
        event = parse_json_line(json.dumps(obj))
        assert event is not None
        assert event.level == "INFO"

    def test_json_millis_ts(self):
        obj = {"timestamp": 1705305000000, "level": "warn", "message": "slow"}
        event = parse_json_line(json.dumps(obj))
        assert event is not None
        # Should convert from millis to seconds
        assert event.timestamp < 2e10

    def test_json_alternative_fields(self):
        obj = {"time": "2024-01-15T08:30:00Z", "severity": "CRITICAL", "msg": "down"}
        event = parse_json_line(json.dumps(obj))
        assert event is not None
        assert event.level == "CRITICAL"

    def test_non_json(self):
        assert parse_json_line("not json at all") is None

    def test_json_no_timestamp(self):
        assert parse_json_line('{"level": "info"}') is None


class TestAutoDetector:
    def test_auto_detects_iso(self):
        parser = LogParser()
        lines = [
            "2024-01-15T08:30:00Z ERROR [auth] Token expired",
            "2024-01-15T08:30:01Z INFO [api] OK",
            "2024-01-15T08:30:02Z WARN [db] Slow",
            "2024-01-15T08:30:03Z INFO [api] OK",
            "2024-01-15T08:30:04Z INFO [api] OK",
            "2024-01-15T08:30:05Z DEBUG [cache] Hit",
        ]
        events = parser.parse_lines(lines)
        assert len(events) == 6
        assert parser.detected_format == "iso"

    def test_auto_detects_json(self):
        parser = LogParser()
        lines = [
            json.dumps({"timestamp": f"2024-01-15T08:30:0{i}Z", "level": "info", "message": "ok"})
            for i in range(6)
        ]
        events = parser.parse_lines(lines)
        assert len(events) == 6
        assert parser.detected_format == "json"

    def test_skips_blank_lines(self):
        parser = LogParser()
        events = parser.parse_lines(["", "  ", "\n"])
        assert len(events) == 0

    def test_handles_mixed_formats(self):
        parser = LogParser()
        lines = [
            "2024-01-15T08:30:00Z INFO [api] OK",
            json.dumps({"timestamp": "2024-01-15T08:30:01Z", "level": "info", "message": "ok"}),
        ]
        events = parser.parse_lines(lines)
        assert len(events) == 2


class TestLogEventProperties:
    def test_datetime_utc(self):
        event = parse_iso_line("2024-01-15T08:30:00Z INFO test")
        assert event is not None
        dt = event.datetime_utc
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 8
        assert dt.minute == 30
