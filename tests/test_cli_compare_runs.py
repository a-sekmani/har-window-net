"""Tests for compare_runs CLI: aggregate run configs and test metrics to CSV."""

import csv
import json
from pathlib import Path

import pytest

from har_windownet.cli.compare_runs import main


def test_compare_runs_writes_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Create two run dirs with checkpoint markers; one has config+metrics, one minimal. Check CSV."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Run 1: full config + metrics
    run1 = runs_dir / "exp_norm"
    run1.mkdir()
    (run1 / "best.ckpt").write_bytes(b"dummy")
    (run1 / "config.json").write_text(
        json.dumps({
            "model": "tcn",
            "feature_config": {"features": "norm", "conf_mode": "keep"},
        }),
        encoding="utf-8",
    )
    (run1 / "reports").mkdir()
    (run1 / "reports" / "test_metrics.json").write_text(
        json.dumps({"accuracy": 0.8, "macro_f1": 0.79}),
        encoding="utf-8",
    )

    # Run 2: only checkpoint (no config, no metrics)
    run2 = runs_dir / "legacy_run"
    run2.mkdir()
    (run2 / "last.ckpt").write_bytes(b"dummy")

    out_csv = tmp_path / "compare.csv"
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["compare_runs", "--runs", str(runs_dir), "--out", str(out_csv)]
        main()
    finally:
        sys.argv = old_argv

    assert out_csv.exists()
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    by_run = {r["run"]: r for r in rows}
    assert "exp_norm" in by_run
    assert by_run["exp_norm"]["model"] == "tcn"
    assert by_run["exp_norm"]["features"] == "norm"
    assert by_run["exp_norm"]["accuracy"] == "0.8"
    assert by_run["exp_norm"]["macro_f1"] == "0.79"
    assert by_run["legacy_run"]["model"] == "tcn"
    assert by_run["legacy_run"]["features"] == "raw"
    assert by_run["legacy_run"]["accuracy"] == ""
    assert by_run["legacy_run"]["macro_f1"] == ""

    out = capsys.readouterr().out
    assert "2 runs" in out or "2" in out


def test_compare_runs_empty_dir(tmp_path: Path) -> None:
    """Runs dir with no run subdirs yields empty CSV."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    out_csv = tmp_path / "out.csv"
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["compare_runs", "--runs", str(runs_dir), "--out", str(out_csv)]
        main()
    finally:
        sys.argv = old_argv
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 0


def test_compare_runs_skips_non_run_subdirs(tmp_path: Path) -> None:
    """Subdirs without best.ckpt/last.ckpt are skipped."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "notes.txt").write_text("x")
    (runs_dir / "real_run").mkdir()
    (runs_dir / "real_run" / "best.ckpt").write_bytes(b"x")
    out_csv = tmp_path / "out.csv"
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["compare_runs", "--runs", str(runs_dir), "--out", str(out_csv)]
        main()
    finally:
        sys.argv = old_argv
    with open(out_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["run"] == "real_run"
