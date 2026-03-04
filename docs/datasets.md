# Datasets

This document describes how to build and validate datasets from NTU RGB+D 120, Custom10 (folder-per-activity), and Edge17 (JSONL) sources.

## Build dataset from NTU

```bash
python -m har_windownet.cli.build_dataset --source <path_to_ntu> --out data_out/ntu120_windows [--projection rgb|depth|3d] [--seed 42] [--window-size 30] [--stride 15] [--export-samples 100]
```

| Option | Description |
|--------|-------------|
| `--source` | Path to the NTU folder containing `*.skeleton` or `*.skeleton.npy`. Use a relative path (e.g. `har_windownet/skeleton_files`) or absolute path. |
| `--out` | Output directory (e.g. `data_out/ntu120_windows`). |
| `--projection` | **rgb** (default when available), **depth**, or **3d**. Use rgb/depth for stable 2D normalization; 3d is unstable across samples. |
| `--window-size`, `--stride` | Frames per window (default 30) and stride between windows (default = window-size). Use a smaller stride for overlapping windows. |
| `--seed` | Random seed for 80/10/10 split (default 42). |
| `--export-samples` | Number of windows to export under `samples/` as JSON (0 = skip). |
| `--no-skip-missing` | Do not skip samples from the official NTU missing-skeletons list. |

If you see **"No NTU samples found"**, ensure `--source` points to a directory that contains `*.skeleton` or `*.skeleton.npy` files (e.g. `S001C002P003R002A013.skeleton`).

## Build dataset from Custom10

For a custom dataset with one folder per activity, each containing clip files:

```bash
python -m har_windownet.cli.build_dataset \
  --dataset custom10 \
  --source <path_to_root_folder> \
  --out data_out/custom10 \
  [--projection rgb|depth|3d] [--window-size 30] [--stride 30] [--fps 30] [--img-w 1920] [--img-h 1080] [--export-samples 20]
```

- **`--source`** must be the directory that contains the label subfolders (e.g. if you have `datasets/A001_WALKING/`, `datasets/A002_SITTING/`, then `--source datasets` from project root). This path is not inside the library; it is your own data folder.
- Each **subfolder** name must match **`A001_*`**, **`001_*`**, or **`A1_*`** / **`A43_*`** (1–3 digits, zero-padded to A001, A043, etc.). Inside each subfolder put **`.json`**, **`.npy`**, or **`.skeleton`** (NTU text) clip files (one file = one clip).
- **`.skeleton`** (NTU text format): use **`--projection rgb`** (default), **`depth`**, or **`3d`** for 2D normalization. RGB/depth use Kinect dimensions (1920×1080 / 512×424); tracking state → confidence (2→1.0, 1→0.5, 0→0.0). Keypoints are sanitized (NaN/Inf replaced and clipped to [0,1]).
- **JSON/NPY** clips must include a **`keypoints`** array shape `[T][K][3]` with K=17 or 25; optional `fps`, `format` (`coco17_norm` or `coco17_pixel`), `img_w`, `img_h`.

If you see **"No Custom10 clips found"**, the error message lists the subfolders found and why they were skipped. See [Expected Custom10 source layout](#expected-custom10-source-layout) below.

## Build dataset from Edge17 (JSONL)

For skeleton data from Edge pose estimation (`.skeleton.jsonl` files, COCO-17, normalized):

```bash
python -m har_windownet.cli.build_dataset \
  --dataset edge17 \
  --source <path_to_jsonl_files> \
  --out data_out/edge17 \
  [--window-size 30] [--stride 15] [--export-samples 20]
```

- **`--source`**: Directory containing `.skeleton.jsonl` files (recursive search).
- Edge17 files are already COCO-17 normalized (0..1), so no projection option is needed.
- **Label**: Extracted from `action_id` in the meta line (e.g., `"action_id": "A001"`).
- **Split**: 80/10/10 by clip_id (filename) to prevent data leakage.

## Validate outputs

```bash
python -m har_windownet.cli.validate_dataset --data data_out/ntu120_windows
```

The same command works for Custom10 or Edge17 output: `--data data_out/custom10` or `--data data_out/edge17`.

Validation checks:

- Presence of `label_map.json` and `splits/` (train/val/test parquet files).
- Every row in each split: required fields present, `keypoints` shape **(T, 17, 3)** where T is `window_size` (from `dataset_meta.json` if present, else 30), no NaN/Inf, x/y/conf in **[0, 1]**, `label` in `label_map`, valid UUIDs for `id` and `session_id`, `label_source` = `"dataset"`.

---

## Expected NTU source layout

- **Option 1 (text files):** A folder with files named like `S001C002P003R002A013.skeleton` (official NTU format).
- **Option 2 (NumPy from parser):** A folder (e.g. `raw_npy/`) with files like `S001C002P003R002A013.skeleton.npy` after running [FesianXu/NTU_RGBD120_Parser_python](https://github.com/FesianXu/NTU_RGBD120_Parser_python). Each `.npy` file, when loaded, yields a dict with:
  - `skel_body0`: shape `(nframe, 25, 3)` (3D coordinates)
  - `rgb_body0`: (if present) 2D RGB projection
  - `nbodys`, `njoints`, etc.

Samples listed in [NTU_RGBD120_samples_with_missing_skeletons](https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD120_samples_with_missing_skeletons.txt) are excluded automatically.

## Expected Custom10 source layout

The **`--source`** path is the **root folder** that **contains** one subfolder per activity (label). The library does not read from `har_windownet/datasets/`; you use your own folder (e.g. `datasets/` at project root).

**Required structure:**

```
<--source>/
├── A001_WALKING/
│   ├── clip1.json
│   └── clip2.npy
├── A002_SITTING/
│   └── video_01.json
├── 001_JUMPING/          # 001_* is also accepted (normalized to A001)
└── A010_OTHER/
    └── sample.json
```

- **Subfolder name:** `A001_*`, `001_*`, or `A1_*` / `A43_*` (optional `A` + 1–3 digits + `_` + rest; normalized to A001, A002, …). Examples: `A001_WALKING`, `A1_drink_water`, `A43_falling_down`.
- **Inside each subfolder:** **`.json`**, **`.npy`**, and **`.skeleton`** (NTU text) files are read; each file = one clip. For **`.skeleton`**, use `--projection rgb|depth|3d` as for NTU.
- **JSON format:** must contain `"keypoints"`: array of shape `[T][K][3]` with **K = 17** (COCO-17) or **25** (NTU, mapped to 17). Optional: `"fps"`, `"format"` (`coco17_norm` or `coco17_pixel`), `"img_w"`, `"img_h"` (required if format is pixel).

**Why "No Custom10 clips found" often happens:**

1. **Path wrong:** `--source datasets` resolves relative to the current working directory. Use an absolute path if unsure.
2. **Subfolder names:** Names like `walking` or `action_1` do not match. You need **`A` + 1–3 digits + `_` + name** (e.g. `A001_WALKING`, `A1_drink_water`).
3. **No clips inside:** Matched subfolders must contain at least one `.json`, `.npy`, or `.skeleton` file.

## Expected Edge17 source layout

Edge17 `.skeleton.jsonl` files are generated by Edge pose estimation pipelines. Each file contains skeleton data for one clip.

**File format:**

- **Line 1 (meta):** `{"type": "meta", "action_id": "A001", "fps": 30.0, "frame_count": 100, "skeleton_format": "coco17", "coords": "normalized", ...}`
- **Lines 2+ (frames):** `{"type": "frame", "frame_index": 0, "ts_unix_ms": ..., "persons": [{"track_id": 1, "keypoints": [[x,y,conf], ...17 keypoints]}]}`

**Key properties:**

- **Keypoints**: Already COCO-17 format, normalized to [0, 1].
- **Label**: Read from `action_id` in the meta line (e.g., `"A001"`).
- **Track**: Uses `track_id: 1` by default.
- **Frames with empty `persons`** (no detection) are skipped.

**Example directory structure:**

```
source/
├── S001C001P001R001A001.skeleton.jsonl
├── S001C001P001R001A008.skeleton.jsonl
└── subdir/
    └── clip_sample.skeleton.jsonl
```

The builder recursively finds all `.skeleton.jsonl` files under `--source`.
