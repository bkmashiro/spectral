"""
Log file parsers for multiple common formats.

Supports: syslog, JSON lines, Common Log Format (CLF), and generic
timestamp-prefixed lines.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, TextIO

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class LogEvent:
    """A single parsed log event."""
    timestamp: float  # UNIX epoch seconds
    level: str = ""   # e.g. INFO, ERROR, WARN
    source: str = ""  # e.g. syslog tag, JSON logger name
    message: str = ""
    raw: str = ""

    @property
    def datetime_utc(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Individual parsers
# ---------------------------------------------------------------------------

# Syslog: "Apr 13 14:22:01 myhost myapp[1234]: Something happened"
_SYSLOG_RE = re.compile(
    r"^(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+"
    r"(?P<source>\S+?)(?:\[\d+\])?:\s+"
    r"(?P<message>.*)$"
)

# CLF: '127.0.0.1 - frank [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif ..." 200 2326'
_CLF_RE = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+'
    r'\[(?P<timestamp>[^\]]+)\]\s+'
    r'"(?P<method>\w+)\s+(?P<path>\S+)\s+\S+"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\S+)'
)

# ISO-prefixed: "2024-01-15T08:30:00Z ERROR [module] message"
_ISO_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\s+"
    r"(?:(?P<level>DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL)\s+)?"
    r"(?:\[(?P<source>[^\]]+)\]\s+)?"
    r"(?P<message>.*)"
)

_MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_syslog_ts(ts: str, year: int | None = None) -> float:
    """Parse syslog timestamp (no year) into epoch seconds."""
    if year is None:
        year = datetime.now().year
    parts = ts.split()
    month = _MONTH_MAP.get(parts[0], 1)
    day = int(parts[1])
    h, m, s = (int(x) for x in parts[2].split(":"))
    dt = datetime(year, month, day, h, m, s, tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_clf_ts(ts: str) -> float:
    """Parse CLF timestamp like '10/Oct/2000:13:55:36 -0700'."""
    try:
        dt = datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S %z")
    except ValueError:
        dt = datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_iso_ts(ts: str) -> float:
    """Parse ISO 8601 timestamp."""
    ts = ts.strip()
    # Normalize for fromisoformat
    ts = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        # Fallback: try common variations
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(ts, fmt)
                dt = dt.replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse timestamp: {ts!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_syslog_line(line: str) -> LogEvent | None:
    m = _SYSLOG_RE.match(line)
    if not m:
        return None
    return LogEvent(
        timestamp=_parse_syslog_ts(m.group("timestamp")),
        source=m.group("source"),
        message=m.group("message"),
        raw=line,
    )


def parse_clf_line(line: str) -> LogEvent | None:
    m = _CLF_RE.match(line)
    if not m:
        return None
    status = int(m.group("status"))
    level = "ERROR" if status >= 500 else "WARN" if status >= 400 else "INFO"
    return LogEvent(
        timestamp=_parse_clf_ts(m.group("timestamp")),
        level=level,
        source=m.group("method"),
        message=f"{m.group('method')} {m.group('path')} {status}",
        raw=line,
    )


def parse_json_line(line: str) -> LogEvent | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    # Try common JSON log field names
    ts_raw = obj.get("timestamp") or obj.get("time") or obj.get("@timestamp") or obj.get("ts")
    if ts_raw is None:
        return None
    if isinstance(ts_raw, (int, float)):
        ts = float(ts_raw)
        # Detect millisecond timestamps
        if ts > 1e12:
            ts /= 1000.0
    else:
        ts = _parse_iso_ts(str(ts_raw))
    level = str(obj.get("level", obj.get("severity", ""))).upper()
    source = str(obj.get("logger", obj.get("source", obj.get("module", ""))))
    message = str(obj.get("message", obj.get("msg", "")))
    return LogEvent(timestamp=ts, level=level, source=source, message=message, raw=line)


def parse_iso_line(line: str) -> LogEvent | None:
    m = _ISO_RE.match(line)
    if not m:
        return None
    return LogEvent(
        timestamp=_parse_iso_ts(m.group("timestamp")),
        level=(m.group("level") or "").upper(),
        source=m.group("source") or "",
        message=m.group("message") or "",
        raw=line,
    )


# ---------------------------------------------------------------------------
# Auto-detecting multi-format parser
# ---------------------------------------------------------------------------

_PARSERS = [parse_json_line, parse_iso_line, parse_syslog_line, parse_clf_line]


@dataclass
class LogParser:
    """Auto-detecting log parser that tries multiple formats."""
    detected_format: str | None = None
    year_hint: int | None = None
    _format_votes: dict = field(default_factory=lambda: {
        "json": 0, "iso": 0, "syslog": 0, "clf": 0
    })
    _format_names: dict = field(default_factory=lambda: {
        "json": parse_json_line,
        "iso": parse_iso_line,
        "syslog": parse_syslog_line,
        "clf": parse_clf_line,
    })

    def parse_line(self, line: str) -> LogEvent | None:
        line = line.rstrip("\n\r")
        if not line:
            return None

        # If we've detected a format, try it first
        if self.detected_format:
            result = self._format_names[self.detected_format](line)
            if result:
                return result

        # Try all parsers
        for name, parser in self._format_names.items():
            result = parser(line)
            if result:
                self._format_votes[name] = self._format_votes.get(name, 0) + 1
                # Auto-detect after 5 successful parses of same format
                total = sum(self._format_votes.values())
                if total >= 5 and self._format_votes[name] / total > 0.8:
                    self.detected_format = name
                return result
        return None

    def parse_file(self, fh: TextIO) -> Iterator[LogEvent]:
        for line in fh:
            event = self.parse_line(line)
            if event is not None:
                yield event

    def parse_lines(self, lines: list[str]) -> list[LogEvent]:
        results = []
        for line in lines:
            event = self.parse_line(line)
            if event is not None:
                results.append(event)
        return results
