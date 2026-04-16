"""
Command-line interface for Spectral.
"""

from __future__ import annotations

import argparse
import sys

import json

from .analysis import analyze_signal, detect_anomalies, compute_summary
from .generator import generate_log_lines, generate_json_log_lines
from .parser import LogParser
from .signal import events_to_signals, group_by_level, group_by_source, group_all
from .visualize import (
    render_full_report,
    render_spectrogram,
    render_spectrum,
    render_waveform,
    render_anomaly_timeline,
    render_summary_stats,
)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze a log file."""
    parser = LogParser()

    if args.input == "-":
        events = list(parser.parse_file(sys.stdin))
    else:
        with open(args.input, "r") as fh:
            events = list(parser.parse_file(fh))

    if not events:
        print("No events parsed. Check log format.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(events)} events", file=sys.stderr)
    if parser.detected_format:
        print(f"Detected format: {parser.detected_format}", file=sys.stderr)

    # Choose grouping
    group_fns = {
        "all": group_all,
        "level": group_by_level,
        "source": group_by_source,
    }
    group_fn = group_fns.get(args.group, group_all)

    signals = events_to_signals(
        events,
        group_fn=group_fn,
        bucket_size=args.bucket if args.bucket else None,
    )

    width = args.width

    window_name = getattr(args, "window", "hamming")
    compute_ci = getattr(args, "confidence", False)
    export_path = getattr(args, "export", None)
    show_summary = getattr(args, "summary", False)

    export_records = []

    for signal in signals:
        result = analyze_signal(
            signal, max_peaks=args.peaks,
            window_name=window_name,
            compute_confidence=compute_ci,
            detect_harmonics=True,
        )
        anomalies = detect_anomalies(signal, threshold=args.threshold)
        print()
        print(render_full_report(signal, result, anomalies, width=width))
        if show_summary:
            summary = compute_summary(signal)
            print()
            print(render_summary_stats(summary))
        if args.spectrogram:
            print()
            print(render_spectrogram(signal, width=width))
        print()

        if export_path:
            record = result.to_dict()
            record["anomalies"] = [
                {
                    "window_start": a.window_start,
                    "window_end": a.window_end,
                    "energy_ratio": a.energy_ratio,
                    "z_score": a.z_score,
                    "severity": a.severity,
                    "description": a.description,
                }
                for a in anomalies
            ]
            export_records.append(record)

    if export_path:
        with open(export_path, "w") as fh:
            json.dump(export_records, fh, indent=2)
        print(f"Exported analysis to {export_path}", file=sys.stderr)


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a demo with synthetic data."""
    print("Generating synthetic log data...", file=sys.stderr)

    lines = generate_log_lines(
        duration_hours=args.hours,
        base_rate=args.rate,
        anomaly_at=0.6 if args.anomaly else None,
        seed=4825,
    )
    print(f"Generated {len(lines)} log lines", file=sys.stderr)

    if args.save:
        with open(args.save, "w") as fh:
            fh.write("\n".join(lines))
        print(f"Saved to {args.save}", file=sys.stderr)

    # Parse and analyze
    parser = LogParser()
    events = parser.parse_lines(lines)

    signals = events_to_signals(events, group_fn=group_by_level)

    width = args.width

    for signal in signals:
        result = analyze_signal(signal, max_peaks=8, window_name="hamming")
        anomalies = detect_anomalies(signal, threshold=2.5)
        print()
        print(render_full_report(signal, result, anomalies, width=width))
        summary = compute_summary(signal)
        print()
        print(render_summary_stats(summary))
        print()


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate synthetic logs to stdout."""
    if args.format == "json":
        lines = generate_json_log_lines(
            duration_hours=args.hours,
            base_rate=args.rate,
            anomaly_at=0.6 if args.anomaly else None,
            seed=4825,
        )
    else:
        lines = generate_log_lines(
            duration_hours=args.hours,
            base_rate=args.rate,
            anomaly_at=0.6 if args.anomaly else None,
            seed=4825,
        )
    for line in lines:
        print(line)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="spectral",
        description="Frequency-domain log analysis — find hidden periodicities in your logs.",
    )
    sub = ap.add_subparsers(dest="command")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a log file")
    p_analyze.add_argument("input", help="Log file path or - for stdin")
    p_analyze.add_argument("-g", "--group", default="all",
                           choices=["all", "level", "source"],
                           help="How to group events into signals")
    p_analyze.add_argument("-b", "--bucket", type=float, default=None,
                           help="Bucket size in seconds (auto-detected if omitted)")
    p_analyze.add_argument("-p", "--peaks", type=int, default=10,
                           help="Max spectral peaks to report")
    p_analyze.add_argument("-t", "--threshold", type=float, default=2.5,
                           help="Anomaly detection threshold (energy ratio)")
    p_analyze.add_argument("-w", "--width", type=int, default=72,
                           help="Display width in characters")
    p_analyze.add_argument("--spectrogram", action="store_true",
                           help="Also show spectrogram")
    p_analyze.add_argument("--window", default="hamming",
                           choices=["hamming", "hanning", "blackman", "rectangular"],
                           help="Window function for FFT (default: hamming)")
    p_analyze.add_argument("--export", type=str, default=None, metavar="FILE",
                           help="Export full analysis as JSON to file")
    p_analyze.add_argument("--confidence", action="store_true",
                           help="Compute bootstrap confidence intervals for peaks")
    p_analyze.add_argument("--summary", action="store_true",
                           help="Show summary statistics")
    p_analyze.set_defaults(func=cmd_analyze)

    # demo
    p_demo = sub.add_parser("demo", help="Run demo with synthetic data")
    p_demo.add_argument("--hours", type=float, default=6.0,
                        help="Duration of synthetic data")
    p_demo.add_argument("--rate", type=float, default=15.0,
                        help="Base event rate per minute")
    p_demo.add_argument("--anomaly", action="store_true", default=True,
                        help="Inject an anomaly")
    p_demo.add_argument("--no-anomaly", action="store_false", dest="anomaly")
    p_demo.add_argument("--save", type=str, default=None,
                        help="Save generated logs to file")
    p_demo.add_argument("-w", "--width", type=int, default=72,
                        help="Display width")
    p_demo.set_defaults(func=cmd_demo)

    # generate
    p_gen = sub.add_parser("generate", help="Generate synthetic logs to stdout")
    p_gen.add_argument("--hours", type=float, default=6.0)
    p_gen.add_argument("--rate", type=float, default=15.0)
    p_gen.add_argument("--format", choices=["iso", "json"], default="iso")
    p_gen.add_argument("--anomaly", action="store_true", default=False)
    p_gen.set_defaults(func=cmd_generate)

    args = ap.parse_args(argv)

    if not args.command:
        ap.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
