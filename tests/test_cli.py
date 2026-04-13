"""Tests for the CLI module."""

import os
import sys
import tempfile

import pytest

from spectral.cli import main
from spectral.generator import generate_log_lines


class TestCLIAnalyze:
    def test_analyze_file(self, capsys):
        lines = generate_log_lines(duration_hours=0.5, seed=42)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("\n".join(lines))
            f.flush()
            path = f.name

        try:
            main(["analyze", path, "-g", "all", "-w", "60"])
        finally:
            os.unlink(path)

        captured = capsys.readouterr()
        assert "SPECTRAL ANALYSIS REPORT" in captured.out

    def test_analyze_by_level(self, capsys):
        lines = generate_log_lines(duration_hours=0.5, seed=42)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("\n".join(lines))
            path = f.name

        try:
            main(["analyze", path, "-g", "level", "-w", "60"])
        finally:
            os.unlink(path)

        captured = capsys.readouterr()
        assert "SPECTRAL ANALYSIS REPORT" in captured.out

    def test_analyze_with_spectrogram(self, capsys):
        lines = generate_log_lines(duration_hours=1.0, seed=42)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("\n".join(lines))
            path = f.name

        try:
            main(["analyze", path, "--spectrogram", "-w", "60"])
        finally:
            os.unlink(path)

        captured = capsys.readouterr()
        assert "Spectrogram" in captured.out


class TestCLIDemo:
    def test_demo(self, capsys):
        main(["demo", "--hours", "0.5", "-w", "60"])
        captured = capsys.readouterr()
        assert "SPECTRAL ANALYSIS REPORT" in captured.out

    def test_demo_save(self, capsys, tmp_path):
        outfile = str(tmp_path / "demo.log")
        main(["demo", "--hours", "0.5", "--save", outfile, "-w", "60"])
        assert os.path.exists(outfile)
        with open(outfile) as f:
            content = f.read()
        assert len(content) > 0


class TestCLIGenerate:
    def test_generate_iso(self, capsys):
        main(["generate", "--hours", "0.1", "--format", "iso"])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) > 0
        assert "T" in lines[0]  # ISO timestamp

    def test_generate_json(self, capsys):
        main(["generate", "--hours", "0.1", "--format", "json"])
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert len(lines) > 0
        import json
        json.loads(lines[0])  # Should not raise


class TestCLIEdgeCases:
    def test_no_command(self):
        with pytest.raises(SystemExit):
            main([])

    def test_empty_file(self, capsys):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            path = f.name

        try:
            with pytest.raises(SystemExit):
                main(["analyze", path])
        finally:
            os.unlink(path)
