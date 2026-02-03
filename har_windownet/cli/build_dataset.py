"""CLI: build Phase A dataset from NTU RGB+D 120."""

from __future__ import annotations

import argparse

from har_windownet.datasets.ntu.builder import build_dataset
from har_windownet.datasets.ntu.config import (
    DEFAULT_PROJECTION,
    PROJECTION_DEPTH,
    PROJECTION_RGB,
    PROJECTION_3D,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build Window dataset from NTU RGB+D 120 (Phase A)"
    )
    p.add_argument(
        "--source",
        required=True,
        help="Path to NTU folder (.skeleton or .npy files)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output directory (e.g. data_out/ntu120_windows)",
    )
    p.add_argument(
        "--projection",
        choices=[PROJECTION_RGB, PROJECTION_DEPTH, PROJECTION_3D],
        default=DEFAULT_PROJECTION,
        help="Projection for 2D normalization: rgb (default), depth, or 3d",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Frames per window (default 30)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between windows (default: window-size, no overlap)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for 80/10/10 split (default 42)",
    )
    p.add_argument(
        "--export-samples",
        type=int,
        default=0,
        metavar="N",
        help="Export first N windows to out/samples/ as JSON (default 0)",
    )
    p.add_argument(
        "--no-skip-missing",
        action="store_true",
        help="Do not skip NTU official missing-skeleton samples",
    )
    args = p.parse_args()

    meta = build_dataset(
        args.source,
        args.out,
        projection=args.projection,
        window_size=args.window_size,
        stride=args.stride,
        seed=args.seed,
        skip_missing=not args.no_skip_missing,
        export_samples_count=args.export_samples,
    )
    print(f"Done. train={meta['num_train_windows']} val={meta['num_val_windows']} test={meta['num_test_windows']}")


if __name__ == "__main__":
    main()
