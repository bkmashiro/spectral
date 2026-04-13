"""
Synthetic log generator for demos and testing.

Uses the seed numbers to parameterize periodic patterns, noise levels,
and anomaly injection.

Seed numbers: 4825 2015 7920 6698 2410 5522 6773 7253 8348 2751 3314 5177 4389 9776 4708 9089
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone

SEEDS = [4825, 2015, 7920, 6698, 2410, 5522, 6773, 7253,
         8348, 2751, 3314, 5177, 4389, 9776, 4708, 9089]


def _derive_periods() -> list[float]:
    """Derive periodic intervals from seed numbers."""
    # Use digit sums and modular arithmetic
    periods = []
    for s in SEEDS[:6]:
        # Map to useful period ranges
        digit_sum = sum(int(d) for d in str(s))
        # Use modular mapping to common intervals
        base = s % 3600  # up to 1 hour
        if base < 60:
            base += 60  # at least 1 minute
        periods.append(float(base))
    return periods


def _derive_noise_level() -> float:
    """Derive noise from seeds."""
    # 7920 is a highly composite number, use it
    return (SEEDS[2] % 100) / 100.0 * 0.3  # 0-0.3 noise level


def generate_log_lines(
    duration_hours: float = 24.0,
    base_rate: float = 10.0,
    periods: list[tuple[float, float, str]] | None = None,
    anomaly_at: float | None = None,
    anomaly_duration: float = 300.0,
    anomaly_multiplier: float = 5.0,
    noise_level: float | None = None,
    seed: int | None = None,
    start_time: float | None = None,
) -> list[str]:
    """
    Generate synthetic log lines with embedded periodic patterns.

    Args:
        duration_hours: Total duration of logs.
        base_rate: Base events per minute.
        periods: List of (period_seconds, amplitude, level) tuples.
                 If None, derived from seed numbers.
        anomaly_at: Fraction of duration [0,1] at which to inject anomaly.
        anomaly_duration: Duration of anomaly in seconds.
        anomaly_multiplier: Rate multiplier during anomaly.
        noise_level: Random noise level [0,1]. Derived from seeds if None.
        seed: Random seed for reproducibility.
        start_time: Start epoch. Defaults to 2026-04-13 00:00:00 UTC.

    Returns:
        List of ISO-formatted log lines.
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(SEEDS[0])

    if start_time is None:
        start_time = datetime(2026, 4, 13, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    duration_sec = duration_hours * 3600
    if noise_level is None:
        noise_level = _derive_noise_level()

    if periods is None:
        derived = _derive_periods()
        periods = [
            (derived[0], 0.4, "ERROR"),     # ~825s period errors
            (derived[1], 0.3, "WARN"),      # ~2015s period warnings
            (derived[2], 0.6, "INFO"),      # ~720s period info
            (3600.0, 0.5, "INFO"),          # hourly pattern
            (300.0, 0.2, "DEBUG"),          # 5-min pattern
        ]

    sources = ["auth", "api", "worker", "scheduler", "cache", "db"]
    messages = {
        "ERROR": [
            "Connection refused to upstream",
            "Timeout waiting for response",
            "Disk write failed: I/O error",
            "Authentication token expired",
            "Rate limit exceeded",
        ],
        "WARN": [
            "Slow query detected (>2s)",
            "Memory usage above 80%",
            "Retry attempt 3/5",
            "Certificate expiring in 7 days",
            "Deprecated API endpoint called",
        ],
        "INFO": [
            "Request processed successfully",
            "Health check passed",
            "Cache refreshed",
            "User session started",
            "Batch job completed",
        ],
        "DEBUG": [
            "Query plan optimized",
            "Connection pool stats: 45/50 active",
            "GC pause: 12ms",
            "Config reloaded from file",
        ],
    }

    lines = []
    t = start_time
    end_time = start_time + duration_sec

    while t < end_time:
        # Compute instantaneous rate from all periodic components
        elapsed = t - start_time
        rate = base_rate

        for period, amplitude, _ in periods:
            rate += base_rate * amplitude * (1 + math.sin(2 * math.pi * elapsed / period)) / 2

        # Anomaly injection
        if anomaly_at is not None:
            anomaly_start = start_time + anomaly_at * duration_sec
            if anomaly_start <= t < anomaly_start + anomaly_duration:
                rate *= anomaly_multiplier

        # Add noise
        rate *= (1 + random.gauss(0, noise_level))
        rate = max(0.1, rate)

        # Generate event
        # Pick level weighted by periodic components
        level_weights: dict[str, float] = {"INFO": 1.0, "DEBUG": 0.3, "WARN": 0.1, "ERROR": 0.05}
        for period, amplitude, level in periods:
            phase = (1 + math.sin(2 * math.pi * elapsed / period)) / 2
            level_weights[level] = level_weights.get(level, 0) + amplitude * phase

        # Weighted random choice
        total_w = sum(level_weights.values())
        r = random.random() * total_w
        chosen_level = "INFO"
        cumulative = 0.0
        for lvl, w in level_weights.items():
            cumulative += w
            if r <= cumulative:
                chosen_level = lvl
                break

        source = random.choice(sources)
        msg_list = messages.get(chosen_level, messages["INFO"])
        msg = random.choice(msg_list)

        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        ts = dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
        line = f"{ts} {chosen_level} [{source}] {msg}"
        lines.append(line)

        # Inter-arrival time (exponential)
        interval = 60.0 / rate
        t += random.expovariate(1.0 / interval) if interval > 0 else 0.1

    return lines


def generate_json_log_lines(
    duration_hours: float = 24.0,
    **kwargs,
) -> list[str]:
    """Generate synthetic logs in JSON format."""
    import json

    iso_lines = generate_log_lines(duration_hours=duration_hours, **kwargs)
    json_lines = []
    for line in iso_lines:
        # Parse the ISO line we just generated
        parts = line.split(" ", 3)
        if len(parts) >= 4:
            ts, level, source_bracket, msg = parts
            source = source_bracket.strip("[]")
            obj = {
                "timestamp": ts,
                "level": level.lower(),
                "source": source,
                "message": msg,
            }
            json_lines.append(json.dumps(obj))
    return json_lines
