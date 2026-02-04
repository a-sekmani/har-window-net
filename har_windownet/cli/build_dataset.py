"""CLI: build Phase A dataset from NTU RGB+D 120 or Custom10."""

from __future__ import annotations

import argparse

from har_windownet.datasets.ntu.builder import build_dataset as build_dataset_ntu
from har_windownet.datasets.ntu.config import (
    DEFAULT_PROJECTION,
    PROJECTION_DEPTH,
    PROJECTION_RGB,
    PROJECTION_3D,
)
from har_windownet.datasets.custom10.builder import build_dataset_custom10


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build Window dataset from NTU RGB+D 120 or Custom10 (Phase A)"
    )
    p.add_argument(
        "--dataset",
        choices=["ntu", "custom10"],
        default="ntu",
        help="Dataset adapter: ntu (default) or custom10",
    )
    p.add_argument(
        "--source",
        required=True,
        help="Path to source folder (NTU: .skeleton/.npy; Custom10: label subfolders with .json/.npy/.skeleton)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output directory (e.g. data_out/ntu120_windows or data_out/custom10)",
    )
    p.add_argument(
        "--projection",
        choices=[PROJECTION_RGB, PROJECTION_DEPTH, PROJECTION_3D],
        default=DEFAULT_PROJECTION,
        help="[NTU/Custom10 .skeleton] Projection for 2D normalization: rgb (default), depth, or 3d",
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
        help="[NTU] Do not skip NTU official missing-skeleton samples",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="[Custom10] FPS when not in clip (default 30.0)",
    )
    p.add_argument(
        "--img-w",
        type=int,
        default=1920,
        help="[Custom10] Image width for pixel normalization (default 1920)",
    )
    p.add_argument(
        "--img-h",
        type=int,
        default=1080,
        help="[Custom10] Image height for pixel normalization (default 1080)",
    )
    args = p.parse_args()

    if args.dataset == "custom10":
        meta = build_dataset_custom10(
            args.source,
            args.out,
            window_size=args.window_size,
            stride=args.stride,
            fps=args.fps,
            img_w=args.img_w,
            img_h=args.img_h,
            projection=args.projection,
            seed=args.seed,
            export_samples_count=args.export_samples,
        )
    else:
        meta = build_dataset_ntu(
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
