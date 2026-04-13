# Spectral

Frequency-domain log analysis. Treats log event streams as signals and applies FFT to reveal hidden periodic patterns, cyclic anomalies, and rhythmic behaviors buried in your logs.

## The Problem

Logs contain temporal patterns that are invisible to `grep` and hard to spot by eye:
- An error that fires every 3600 seconds (misfiring cron job)
- A gradual daily memory climb with a 24-hour cycle
- A 7-day traffic pattern from weekday/weekend differences
- A sudden shift in event frequency indicating a deployment or failure

These are **frequency-domain** problems. Spectral solves them with signal processing.

## How It Works

1. **Parse** — Auto-detects log format (ISO 8601, JSON lines, syslog, Common Log Format)
2. **Signal** — Converts event timestamps into time-bucketed count signals, grouped by level, source, or custom key
3. **FFT** — Applies Fast Fourier Transform (pure Python, no numpy) to decompose each signal into frequency components
4. **Detect** — Finds spectral peaks (periodicities) and runs sliding-window anomaly detection via spectral energy comparison
5. **Visualize** — Renders frequency spectra, waveforms, spectrograms, and anomaly timelines in the terminal using Unicode block characters

## Installation

```bash
# From the project directory
pip install -e .

# Or just run directly
python -m spectral --help
```

No external dependencies. Pure Python 3.10+ with only standard library modules (`cmath`, `math`, `json`, `re`, `argparse`).

## Usage

### Analyze a log file

```bash
# Basic analysis — all events as one signal
spectral analyze /var/log/syslog

# Group by log level (separate spectrum for ERROR, WARN, INFO, etc.)
spectral analyze app.log -g level

# Group by source module
spectral analyze app.log -g source

# Include spectrogram visualization
spectral analyze app.log --spectrogram

# Custom bucket size (30-second buckets)
spectral analyze app.log -b 30

# Read from stdin
cat logs.json | spectral analyze -
```

### Run the demo

```bash
# Built-in demo with synthetic data
spectral demo

# Or run the full demo script
python demo/run_demo.py
```

### Generate synthetic logs

```bash
# Generate ISO-format logs
spectral generate --hours 24 --format iso > test.log

# Generate JSON-format logs with anomaly
spectral generate --hours 12 --format json --anomaly > test.json
```

## Example Output

```
========================================================================
  SPECTRAL ANALYSIS REPORT: ERROR
========================================================================

  Signal: ERROR  (bucket=30.0s, span=6.0h)
  ▁▁▂▁▁▃▁▁▂▁▁▄▁▁▂▁▁▃▁▁▂▁▁▅▁▁▂▁▁▃▁▁▂▁▁▆▁▁████▇▃▁▁▂▁▁▃▁▁

=== Spectral Analysis: ERROR ===
  Samples: 720, Sample rate: 0.0333 Hz
  DC (mean rate): 2.34
  Spectral energy: 45.12
  Detected periodicities (3):
    #1: period=1.0h (hourly), magnitude=8.45
    #2: period=12.0m, magnitude=3.21
    #3: period=5.0m (every 5 minutes), magnitude=1.87

  Anomaly Timeline (1 detected)
  ┌────────────────────────────────────────────────────────────────────────┐
  │                                          ████████                     │
  └────────────────────────────────────────────────────────────────────────┘
  [    HIGH] 14:32:00 - Spectral energy 3.8x baseline in 1920s window
```

## Architecture

```
src/spectral/
├── __init__.py      # Package metadata
├── __main__.py      # python -m spectral entry point
├── parser.py        # Multi-format log parser with auto-detection
├── signal.py        # Event-to-time-series conversion
├── analysis.py      # Pure-Python FFT, peak detection, anomaly detection
├── visualize.py     # Terminal rendering (Unicode block charts)
├── generator.py     # Synthetic log generator (seed-parameterized)
└── cli.py           # Command-line interface

tests/
├── test_parser.py     # 20 tests — format detection, edge cases
├── test_signal.py     # 11 tests — bucketing, grouping, padding
├── test_analysis.py   # 14 tests — FFT correctness, peak detection, anomalies
├── test_visualize.py  # 8 tests — rendering output validation
├── test_generator.py  # 9 tests — determinism, parseability, end-to-end
└── test_cli.py        # 7 tests — CLI subcommands, edge cases

demo/
└── run_demo.py      # Full interactive demo with commentary
```

## Key Design Decisions

**Pure-Python FFT**: Uses Cooley-Tukey radix-2 decimation-in-time, implemented in ~15 lines with `cmath`. No numpy needed. Perfectly adequate for log analysis where N is typically 512-4096.

**Auto-detecting parser**: Tries all format parsers on each line, tracks hit rates, and locks onto the dominant format after 5 successful parses. Handles mixed-format files gracefully.

**Seed-parameterized generator**: The 16 seed numbers (4825, 2015, 7920, ...) derive periodic intervals, noise levels, and event distributions for the synthetic test data via modular arithmetic and digit-sum mappings.

**Sliding-window anomaly detection**: Computes spectral energy in overlapping windows and flags windows where energy exceeds a configurable multiple of the median baseline. This catches both sustained anomalies and brief spikes.

## Running Tests

```bash
cd spectral
python -m pytest tests/ -v
```
