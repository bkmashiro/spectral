#!/usr/bin/env python3
"""
Spectral Demo - Frequency-domain log analysis in action.

This demo:
1. Generates synthetic logs with known periodic patterns and an anomaly
2. Parses them through the auto-detecting parser
3. Converts to time-series signals grouped by log level
4. Performs FFT analysis to detect periodicities
5. Runs anomaly detection via sliding-window spectral energy
6. Renders terminal visualizations of all findings

Run: python demo/run_demo.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spectral.generator import generate_log_lines, SEEDS
from spectral.parser import LogParser
from spectral.signal import events_to_signals, group_by_level, group_all
from spectral.analysis import analyze_signal, detect_anomalies
from spectral.visualize import (
    render_full_report,
    render_spectrogram,
    render_waveform,
)


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print()


def main():
    section("SPECTRAL - Frequency-Domain Log Analysis Demo")
    print("  Seed numbers: ", " ".join(str(s) for s in SEEDS))
    print()

    # --- Step 1: Generate synthetic logs ---
    section("Step 1: Generating Synthetic Logs")
    print("  Creating 6 hours of logs with:")
    print("    - Hourly ERROR pattern (period ~3600s)")
    print("    - 5-minute DEBUG heartbeat (period ~300s)")
    print("    - ~12-minute INFO cycle (period ~720s)")
    print("    - Anomaly injected at 60% mark (sudden 5x spike)")
    print()

    lines = generate_log_lines(
        duration_hours=6.0,
        base_rate=15.0,
        anomaly_at=0.6,
        anomaly_duration=300.0,
        anomaly_multiplier=5.0,
        seed=SEEDS[0],
    )
    print(f"  Generated {len(lines)} log events")
    print()
    print("  Sample lines:")
    for line in lines[:5]:
        print(f"    {line}")
    print("    ...")

    # --- Step 2: Parse ---
    section("Step 2: Parsing with Auto-Detection")
    parser = LogParser()
    events = parser.parse_lines(lines)
    print(f"  Parsed {len(events)} events")
    print(f"  Detected format: {parser.detected_format}")

    level_counts = {}
    for e in events:
        level_counts[e.level] = level_counts.get(e.level, 0) + 1
    print("  Level distribution:")
    for level, count in sorted(level_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(50, count // (len(events) // 50 + 1))
        print(f"    {level:>8}: {count:>5} {bar}")

    # --- Step 3: Aggregate signal (all events) ---
    section("Step 3: All-Events Analysis")
    signals_all = events_to_signals(events, group_fn=group_all, bucket_size=30.0)

    for sig in signals_all:
        result = analyze_signal(sig, max_peaks=8)
        anomalies = detect_anomalies(sig, threshold=2.5)
        print(render_full_report(sig, result, anomalies, width=70))

    # --- Step 4: Per-level analysis ---
    section("Step 4: Per-Level Spectral Analysis")
    signals_level = events_to_signals(events, group_fn=group_by_level, bucket_size=30.0)

    for sig in signals_level:
        if sig.total < 50:
            continue  # skip sparse signals
        result = analyze_signal(sig, max_peaks=5)
        anomalies = detect_anomalies(sig, threshold=2.5)
        print(render_full_report(sig, result, anomalies, width=70))
        print()

    # --- Step 5: Spectrogram ---
    section("Step 5: Spectrograms")
    for sig in signals_all:
        print(render_spectrogram(sig, width=68, height=20))
        print()

    # --- Summary ---
    section("Summary")
    print("  Spectral analysis can reveal patterns invisible to grep:")
    print()
    print("    - Periodic errors suggest cron jobs, polling loops, or retry cycles")
    print("    - Spectral anomalies highlight sudden behavioral shifts")
    print("    - Per-level decomposition isolates which severities carry the pattern")
    print()
    print("  Usage on real logs:")
    print("    python -m spectral analyze /var/log/syslog -g level")
    print("    python -m spectral analyze app.log --spectrogram")
    print("    cat logs.json | python -m spectral analyze - -g source")
    print()


if __name__ == "__main__":
    main()
