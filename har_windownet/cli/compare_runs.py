"""CLI: aggregate run metrics into compare.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate run configs and test metrics into compare.csv")
    p.add_argument(
        "--runs",
        type=Path,
        default=Path("runs"),
        help="Directory containing run subdirs (default: runs)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <runs>/compare.csv)",
    )
    args = p.parse_args()

    runs_dir = args.runs.resolve()
    if not runs_dir.is_dir():
        raise SystemExit(f"Not a directory: {runs_dir}")

    out_path = args.out or (runs_dir / "compare.csv")
    rows: list[dict[str, str | float | None]] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        has_ckpt = (run_dir / "best.ckpt").exists() or (run_dir / "last.ckpt").exists()
        if not has_ckpt:
            continue

        run_name = run_dir.name
        model = "tcn"
        features = "raw"
        conf_mode = "keep"
        accuracy: float | None = None
        macro_f1: float | None = None

        config_path = run_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                model = cfg.get("model", model)
                fc = cfg.get("feature_config") or {}
                features = fc.get("features", features)
                conf_mode = fc.get("conf_mode", conf_mode)
            except (json.JSONDecodeError, OSError):
                pass

        metrics_path = run_dir / "reports" / "test_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    m = json.load(f)
                accuracy = m.get("accuracy")
                macro_f1 = m.get("macro_f1")
            except (json.JSONDecodeError, OSError):
                pass

        rows.append({
            "run": run_name,
            "model": model,
            "features": features,
            "conf_mode": conf_mode,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "model", "features", "conf_mode", "accuracy", "macro_f1"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} runs to {out_path}")


if __name__ == "__main__":
    main()
