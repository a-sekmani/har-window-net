# HAR-WindowNet

A training workspace for building a data pipeline and ML model that uses the same **Window** format as the cloud system.

## Goals

1. Use **NTU RGB+D 120** skeleton/keypoints data.
2. Convert each sample into **Windows** matching the cloud **Contract**.
3. Train a model to classify activity from Window (keypoint sequences).
4. Export a model for inference in the cloud system.

## Phase A: Data Contract + Dataset Adapter

Phase A deliverables:

- **Window Contract**: Fixed schema (JSON Schema / Pydantic) compatible with the cloud.
- **NTU Adapter**: Read NTU skeleton data, build windows (window_size=30, fps=30).
- **Dataset Builder**: train/val/test splits, metadata, label_map, statistics.
- **Validation Suite**: Verify windows match the cloud format 100%.
- **Sample Export**: A small set of windows for manual checks or cloud upload.

---

## Environment Setup

From the project root, create a virtual environment, activate it, and install the project (with dev tools for tests):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional: install without dev tools (no pytest/ruff): `pip install -e .`  
Check: `python -c "import har_windownet; print(har_windownet.__version__)"`

---

## Usage

### Build dataset from NTU

```bash
python -m har_windownet.cli.build_dataset --source <path_to_ntu> --out data_out/ntu120_windows [--projection rgb|depth|3d] [--seed 42] [--window-size 30] [--stride 15] [--export-samples 100]
```

- `--source`: Path to the NTU folder (files `.skeleton` or `.skeleton.npy`). Use a **relative** path (e.g. `har_windownet/skeleton_files`) or **absolute** path (e.g. `/Users/.../skeleton_files`). The builder discovers only `*.skeleton.npy` and `*.skeleton` under this directory.
- `--out`: Output directory (e.g. `data_out/ntu120_windows`).
- `--projection`: **rgb** (default when available), **depth**, or **3d**. Use rgb/depth for stable 2D normalization; 3d is unstable across samples.
- `--window-size`, `--stride`: Frames per window (default 30) and stride between windows (default = window-size). Use a smaller stride for overlapping windows.
- `--seed`: Random seed for 80/10/10 split (default 42).
- `--export-samples`: Number of windows to export under `samples/` as JSON (optional; 0 = skip).
- `--no-skip-missing`: Do not skip samples from the official NTU missing-skeletons list.

If you see **"No NTU samples found"**, ensure `--source` points to a directory that contains `*.skeleton` or `*.skeleton.npy` files (e.g. `S001C002P003R002A013.skeleton`).

### Build dataset from Custom10

For a custom dataset (one folder per activity, each containing clip files):

```bash
python -m har_windownet.cli.build_dataset \
  --dataset custom10 \
  --source <path_to_root_folder> \
  --out data_out/custom10 \
  [--projection rgb|depth|3d] [--window-size 30] [--stride 30] [--fps 30] [--img-w 1920] [--img-h 1080] [--export-samples 20]
```

### Build dataset from Edge17 (JSONL)

For skeleton data from Edge pose estimation (`.skeleton.jsonl` files, COCO-17, normalized):

```bash
python -m har_windownet.cli.build_dataset \
  --dataset edge17 \
  --source <path_to_jsonl_files> \
  --out data_out/edge17 \
  [--window-size 30] [--stride 15] [--export-samples 20]
```

- **`--source`**: Directory containing `.skeleton.jsonl` files (recursive search).
- **Edge17 files** are already COCO-17 normalized (0..1), so no projection option is needed.
- **Label**: Extracted from `action_id` in the meta line (e.g., `"action_id": "A001"`).
- **Split**: 80/10/10 by clip_id (filename) to prevent data leakage.

- **`--source`** must be the **directory that contains** the label subfolders (e.g. if you have `datasets/A001_WALKING/`, `datasets/A002_SITTING/`, then `--source datasets` from project root). This path is **not** inside the library; it is your own data folder (any name: `datasets`, `my_data`, etc.).
- Each **subfolder** name must match **`A001_*`**, **`001_*`**, or **`A1_*`** / **`A43_*`** (1–3 digits, zero-padded to A001, A043, etc.). Inside each subfolder put **`.json`**, **`.npy`**, or **`.skeleton`** (NTU text) clip files (one file = one clip).
- **`.skeleton`** (NTU text format): read like the NTU adapter; use **`--projection rgb`** (default), **`depth`**, or **`3d`** for 2D normalization. RGB/depth use Kinect dimensions (1920×1080 / 512×424); tracking state → confidence (2→1.0, 1→0.5, 0→0.0). Keypoints are sanitized (NaN/Inf replaced and clipped to [0,1]) so validation and training always see valid values.
- **JSON/NPY** clips must include a **`keypoints`** array shape `[T][K][3]` with K=17 or 25; optional `fps`, `format` (`coco17_norm` or `coco17_pixel`), `img_w`, `img_h`.

If you see **"No Custom10 clips found"**, the error message will list the subfolders found and why they were skipped (name not matched, or no .json/.npy/.skeleton inside). See **Expected Custom10 source layout** below.

### Validate outputs

```bash
python -m har_windownet.cli.validate_dataset --data data_out/ntu120_windows
```

The same command works for Custom10 output: `--data data_out/custom10`.

Validation checks:

- Presence of `label_map.json` and `splits/` (train/val/test parquet files).
- Every row in each split: required fields present, `keypoints` shape **(30, 17, 3)**, no NaN/Inf, x/y/conf in **[0, 1]**, `label` in `label_map`, valid UUIDs for `id` and `session_id`, `label_source` = `"dataset"`.

---

## Expected NTU source layout

- **Option 1 (text files)**: A folder with files named like `S001C002P003R002A013.skeleton` (official NTU format).
- **Option 2 (NumPy from parser)**: A folder (e.g. `raw_npy/`) with files like `S001C002P003R002A013.skeleton.npy` after running [FesianXu/NTU_RGBD120_Parser_python](https://github.com/FesianXu/NTU_RGBD120_Parser_python). Each `.npy` file, when loaded, yields a dict with:
  - `skel_body0`: shape `(nframe, 25, 3)` (3D coordinates)
  - `rgb_body0`: (if present) 2D RGB projection
  - `nbodys`, `njoints`, etc.

Samples listed in [NTU_RGBD120_samples_with_missing_skeletons](https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD120_samples_with_missing_skeletons.txt) are excluded automatically.

---

## Expected Custom10 source layout

The **`--source`** path is the **root folder** that **contains** one subfolder per activity (label). The library does **not** read from `har_windownet/datasets/`; you use your own folder (e.g. `datasets/` at project root).

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

1. **Path wrong:** `--source datasets` resolves relative to the current working directory. If you run from project root and the folder is `./datasets`, it must exist and contain the subfolders. Use an absolute path if unsure.
2. **Subfolder names:** Names like `walking` or `action_1` do **not** match. You need **`A` + 1–3 digits + `_` + name** (e.g. `A001_WALKING`, `A1_drink_water`, `A43_falling_down`).
3. **No clips inside:** Matched subfolders must contain at least one `.json`, `.npy`, or `.skeleton` file.

---

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

---

## Window Contract

A window represents: one person (track_id), one camera, a fixed-length segment of N frames.

| Field            | Type     | Description                          |
|------------------|----------|--------------------------------------|
| id               | UUID str | Unique window id                     |
| device_id        | str      | e.g. `ntu-offline`                   |
| camera_id        | str      | e.g. `ntu-cam`                       |
| session_id       | UUID str | Per video/sample                     |
| track_id         | int      | Usually 1                            |
| ts_start_ms, ts_end_ms | int | Start time (e.g. 0 offline); end = **floor**((window_size−1)×(1000/fps)) + ts_start_ms — see below |
| fps              | float    | 30.0 (float for cloud compatibility)                  |
| window_size      | int      | 30                                   |
| mean_pose_conf   | float    | Mean keypoint confidence (from tracking_state or default) |
| label            | str      | Activity label from NTU              |
| label_source     | `"dataset"` | Fixed                             |
| created_at       | ISO str  | Build timestamp                      |
| keypoints        | array    | Shape [T][K][3] = [30][17][3], (x, y, conf); x,y in [0..1] |
| source_body_id   | int?     | NTU body index used (0 = dominant); for multi-person traceability |

T=30 frames, K=17 keypoints (COCO-17), each (x, y, confidence).

---

## Conversion choices (NTU → Window)

### Projection (3D → 2D)

NTU 3D (x,y,z) is in camera space, not pixels. Normalizing by “scene range” is **unstable** across samples (camera distance varies). Prefer:

- **RGB projection** when available: use `colorX/colorY` (`.skeleton`) or `rgb_body0` (`.npy`), then normalize by **1920×1080**.
- **Depth projection** otherwise: use `depthX/depthY`, then normalize by **512×424**.
- **3D** only as fallback: use (x,y) from 3D and normalize by scene bounds (documented as unstable).

The pipeline supports an explicit **`--projection rgb|depth|3d`** option (default: **rgb** when available). See `har_windownet/datasets/ntu/config.py` and `preprocess.py`.

### Confidence

Do not use a fixed `1.0`. Use Kinect V2 **trackingState** per joint (from `.skeleton`):

- **2 (Tracked)** → conf = 1.0  
- **1 (Inferred)** → conf = 0.5  
- **0 (NotTracked)** → conf = 0.0  

`mean_pose_conf` is computed from these values. For `.npy` (no tracking state), default 1.0 is used. See `har_windownet/datasets/ntu/mapping.tracking_state_to_confidence`.

### Timestamps (offline NTU)

For offline NTU, timestamps are synthetic but **consistent**. Use **floor** (integer truncation) so training and cloud agree; do **not** mix with `round()`.

- `ts_start_ms`: 0 for the first window, then `stride × (1000/fps)` per subsequent window.
- `ts_end_ms = ts_start_ms + floor((window_size - 1) * (1000 / fps))`  
  Example: window_size=30, fps=30, ts_start_ms=0 → 29×(1000/30)=966.66… → **966** (not 967).

Helper: `har_windownet.contracts.window.ts_end_ms_from_window(window_size, fps, ts_start_ms)`.

### Mapping (25 → 17)

COCO-17 order is fixed; same order in training and cloud. Nose/ears/eyes have no direct NTU joint and use **Head(3)** as proxy; their confidence can be 0 if Head is not tracked (some actions depend on head). See `har_windownet/datasets/ntu/mapping.py`.

### Multi-person

NTU can have more than one body per frame. Policy is **fixed**:

- **Dominant body**: choose by **most tracked joints** (`tracking_state == 2`), or by **closest z** (mean spine-base z).  
  Use `har_windownet.datasets.ntu.reader.select_dominant_body(bodies, policy="most_tracked"|"closest_z")`.
- Store **`source_body_id`** (body index used) in each window and/or in `dataset_meta.json` for traceability.

---

## Phase A output layout

After `build_dataset` (NTU or Custom10), the output directory has the same structure:

```
data_out/ntu120_windows/
  label_map.json          # id_to_name, label_to_id (A001..A120), num_classes
  dataset_meta.json      # source_dir, projection, window_size, stride, fps, seed, counts
  splits/
    train.parquet
    val.parquet
    test.parquet
  stats/                  # always written
    class_counts.json    # window count per label (e.g. "A001": 123)
    pose_conf_hist.json  # mean_pose_conf: bin_edges, counts (10 bins), min, max, mean
  samples/                # only if --export-samples N > 0
    window_00000.json
    window_00001.json
    ...
```

Parquet files are standard Arrow/Parquet (signature PAR1). Each row is one window with columns matching the Window Contract; `keypoints` is stored as nested list `[30][17][3]`.

---

## Phase B: Training, Evaluation, Export, Inference

Phase B consumes Phase A output only (no window building). It adds training (TCN/GRU), evaluation (accuracy, macro-F1, confusion matrix, per-class CSV), ONNX export, and offline inference.

## Phase C: Feature Pipeline and Run Config

Phase C adds a **feature pipeline** (normalization, velocity, angles) and **run configuration**:

- **Feature options** (train/eval/export): `raw` (baseline 51-D), `norm`, `vel`, `angles`, `combo`; `conf_mode`: `keep` (x,y,conf) or `drop` (x,y only); `norm_center` / `norm_scale` (e.g. `auto`, `midhip`, `shoulders`).
- **Run config**: Each training run writes `config.json` in the run directory (data path, model, hyperparameters, `feature_config`). Eval can load config from the run dir so `--data` is optional when the run was trained with config.
- **Checkpoints** store `input_features` and `feature_config`; export and inference use them so ONNX and offline inference apply the same transforms when needed.
- **Compare runs**: CLI to aggregate all runs under a directory into a single `compare.csv` (run name, model, features, accuracy, macro_f1).

**Dependencies**:

- Base: `pip install -e .` (torch, scikit-learn, matplotlib, etc.).
- ONNX export: `pip install -e ".[export]"` (installs `onnx`, `onnxscript`). Export uses **opset 18** to avoid version-conversion failures (requesting 14 caused downgrade errors with current PyTorch/ONNX).
- Inference CLI: requires `onnxruntime`. On **Python 3.14** there may be no wheel — use `pip install onnxruntime` if available, or Python 3.11/3.12 with `pip install -e ".[export,export-inference]"`.

After `pip install -e .`, these entry points are available: `har-windownet-build-dataset`, `har-windownet-validate-dataset`, `har-windownet-train`, `har-windownet-eval`, `har-windownet-export-model`, `har-windownet-compare-runs`. You can also run any CLI as `python -m har_windownet.cli.<module>` (e.g. `python -m har_windownet.cli.train`).

### Training steps (end-to-end)

Run these in order from the project root (with `.venv` activated):

1. **Build dataset** (Phase A). Example for Custom10:
   ```bash
   python -m har_windownet.cli.build_dataset \
     --dataset custom10 --source datasets --out data_out/custom10 \
     --projection rgb --window-size 30 --stride 30
   ```
   For NTU, use `--dataset ntu --source <path_to_ntu> --out data_out/ntu120_windows`.

2. **Validate** (optional): `python -m har_windownet.cli.validate_dataset --data data_out/custom10`

3. **Train**: saves `best.ckpt`, `last.ckpt`, and `config.json` under `--out`. Use `--seed` for reproducible runs. Optional: `--features norm|vel|angles|combo`, `--conf-mode keep|drop`, `--norm-center`, `--norm-scale` (see Train section below).
   ```bash
   python -m har_windownet.cli.train \
     --data data_out/custom10 --model tcn --batch-size 64 --epochs 30 --lr 1e-3 \
     --seed 42 --out runs/custom10_tcn_v1
   ```

4. **Eval**: writes `reports/<split>_metrics.json`, `per_class.csv`, `confusion_matrix.png`, and `class_map.csv`. If the checkpoint is in a run dir with `config.json`, `--data` can be omitted (data path is read from config).
   ```bash
   python -m har_windownet.cli.eval \
     --checkpoint runs/custom10_tcn_v1/best.ckpt --split test
   ```
   Or with explicit data: `--data data_out/custom10 --checkpoint runs/custom10_tcn_v1/best.ckpt`.

5. **Export to ONNX** (optional): for deployment. Uses `input_features` and `feature_config` from the checkpoint; `model_meta.json` includes `feature_spec` when present.
   ```bash
   python -m har_windownet.cli.export_model \
     --checkpoint runs/custom10_tcn_v1/best.ckpt --out runs/custom10_tcn_v1/export \
     --data data_out/custom10
   ```

6. **Inference** (optional): run the exported model on a single-window JSON. If the model was trained with a feature pipeline, inference applies the same transforms using `feature_spec` from `model_meta.json`.
   ```bash
   python -m har_windownet.export.inference_onnx \
     --model runs/custom10_tcn_v1/export/model.onnx \
     --window data_out/custom10/samples/window_00000.json
   ```

7. **Compare runs** (optional): aggregate run configs and test metrics into one CSV.
   ```bash
   python -m har_windownet.cli.compare_runs --runs runs --out runs/compare.csv
   ```

### Train

```bash
python -m har_windownet.cli.train --data data_out/ntu120_windows --model tcn --batch-size 64 --epochs 50 --lr 1e-3 --seed 42 --out runs/exp01
```

- `--data` **must** point to a Phase A dataset directory: it must contain `label_map.json` and `splits/train.parquet` (and val/test). If `label_map.json` is missing, you get a clear error: run `build_dataset` with `--out <dir>` first, then use that path as `--data`.
- `--model`: `tcn` (default) or `gru`.
- `--seed` (default 42): random seed for PyTorch and the train DataLoader shuffle, for reproducible training. Use different seeds (e.g. 42, 7, 123) when reporting mean±std across runs.
- **Phase C feature options** (optional):
  - `--features`: `raw` (default, 51-D keypoints), `norm`, `vel`, `angles`, `combo`. `norm` = center/scale normalization; `vel` = add frame-wise velocity; `angles` = add joint angles; `combo` = norm + vel + angles.
  - `--conf-mode`: `keep` (default, x,y,conf) or `drop` (x,y only).
  - `--norm-center`, `--norm-scale`: e.g. `auto`, `midhip`, `shoulders`, `hips` (see `har_windownet.features.transforms`).
- **Training optimizations** (optional):
  - `--class-weights`: Use inverse class frequency weights for imbalanced datasets.
  - `--label-smoothing <float>`: Label smoothing factor (0.0 = no smoothing, 0.1 = recommended).
  - `--lr-scheduler`: Use CosineAnnealingLR scheduler for gradual learning rate decay.
- Saves `best.ckpt` (by validation macro-F1), `last.ckpt`, and **`config.json`** under `--out`. The config holds data path, model name, hyperparameters, and `feature_config` for reproducible eval/export.

### Eval

```bash
python -m har_windownet.cli.eval --checkpoint runs/exp01/best.ckpt --split test
```

Or with explicit data: `--data data_out/ntu120_windows --checkpoint runs/exp01/best.ckpt`.

- **Data**: If the checkpoint lives in a run directory that contains **`config.json`** (from Phase C training), the eval CLI reads the data path from it and **`--data` is optional**. Otherwise you must pass `--data` to the Phase A output directory.
- **Pipeline**: Eval uses the same feature pipeline as training when `feature_config` is present in the run config or in the checkpoint.
- Writes under `reports/`: **`<split>_metrics.json`** (accuracy, macro-F1, per-class), **`per_class.csv`** (class_id, label, precision, recall, f1, support), **`confusion_matrix.png`**, and **`class_map.csv`**. Default reports dir is the checkpoint’s parent directory (`run_dir/reports`).

### Export to ONNX

```bash
python -m har_windownet.cli.export_model --checkpoint runs/exp01/best.ckpt --out runs/exp01/export [--data data_out/ntu120_windows]
```

- Produces **`model.onnx`**, **`model_meta.json`**, and **`label_map.json`** in `--out`. The checkpoint’s **`input_features`** and **`feature_config`** (if present) are used so the ONNX input shape and metadata match the training pipeline. **`model_meta.json`** includes **`input_features`** and **`feature_spec`** (same as `feature_config`) when the model was trained with a feature pipeline.
- Exported model uses **ONNX opset 18**. Using 14 triggered a failed downgrade in the ONNX converter; we use 18 so `onnxruntime` can run the model. You may see a PyTorch warning about `dynamic_axes`; export still completes.

### Offline inference on Window JSON

After exporting the model, run inference (requires `onnxruntime`):

```bash
python -m har_windownet.export.inference_onnx --model runs/exp01/export/model.onnx --window data_out/ntu120_windows/samples/window_00000.json
```

- Pass a **single-window** JSON file (or a list of windows). Returns `pred_label`, `pred_label_id`, and optionally `probs`.
- If **`model_meta.json`** contains **`feature_spec`**, the inference code applies the same feature transforms (normalize, velocity, angles, etc.) to the window keypoints before running the model, so raw Window JSON (30×17×3) works for both baseline and feature-pipeline models.

### Compare runs

Aggregate run directories (each with optional `config.json` and `reports/test_metrics.json`) into a single CSV:

```bash
python -m har_windownet.cli.compare_runs --runs runs --out runs/compare.csv
```

- **`--runs`** (default: `runs`): directory containing one subdirectory per run. Only subdirs that have **`best.ckpt`** or **`last.ckpt`** are included.
- **`--out`**: output CSV path (default: `<runs>/compare.csv`).
- CSV columns: **run**, **model**, **features**, **conf_mode**, **accuracy**, **macro_f1**. Metrics come from `reports/test_metrics.json` when present; config from `config.json`. Old runs without config/metrics still get a row with default model/features and empty accuracy/macro_f1.

### Run directory layout (after train + eval)

A typical run directory (e.g. `runs/exp01`) contains:

- **`config.json`**: data path, model, batch_size, epochs, lr, seed, device, **feature_config** (written at train start).
- **`best.ckpt`**, **`last.ckpt`**: PyTorch checkpoints (model_state_dict, epoch, val_macro_f1, num_classes, model_name, **input_features**, **feature_config**).
- **`reports/`** (after eval): **`<split>_metrics.json`**, **`per_class.csv`**, **`confusion_matrix.png`**, **`class_map.csv`**.

---

## Troubleshooting

| Issue | What to do |
|-------|-------------|
| **No NTU samples found** | `--source` must be a directory containing `*.skeleton` or `*.skeleton.npy`. Use a correct relative path (e.g. `har_windownet/skeleton_files`) or absolute path. |
| **label_map.json not found** (when training) | `--data` must be a Phase A output directory. Run `build_dataset --out <dir>` first, then use that directory as `--data`. |
| **ONNX export: version conversion error** | The project uses **opset 18**; do not lower it to 14 or the ONNX C API converter can fail. Re-export with the default (no change needed in code). |
| **ModuleNotFoundError: onnxruntime** | Install with `pip install onnxruntime` if a wheel exists for your Python version. On Python 3.14 there may be none; use Python 3.11/3.12 for the inference CLI. |
| **ModuleNotFoundError: onnxscript** | Required for `torch.onnx.export`. Install with `pip install -e ".[export]"`. |
| **Eval: Missing --data** | If the run has no `config.json` (e.g. pre–Phase C run), you must pass `--data <path_to_phase_a_output>`. |

---

## Testing

Tests are in the `tests/` directory and use **pytest**.

### Run all tests

With the virtual environment activated and dev dependencies installed:

```bash
pytest
```

From the project root, this discovers and runs all tests under `tests/`.

### Run with verbose output

```bash
pytest -v
```

### Run a specific test file or test

```bash
pytest tests/test_contracts_window.py -v
pytest tests/test_contracts_window.py::test_window_contract_valid -v
```

### Run tests by keyword

```bash
pytest -k "window" -v
pytest -k "mapping" -v
```

### Test coverage (optional)

If you install `pytest-cov`:

```bash
pip install pytest-cov
pytest --cov=har_windownet --cov-report=term-missing
```

### What is tested

- **Contracts**
  - `WindowContract`: validation of id, session_id (UUID), keypoints shape (30, 17, 3), no NaN/Inf, values in [0, 1], serialization/deserialization.
  - `validate_window_dict`: required fields, keypoints shape and value checks.
  - Labels: `build_default_label_map`, `load_label_map`, `save_label_map`, `get_label_id`, `get_label_name`.
- **NTU mapping**
  - `map_ntu_frame_to_coco17`: output shape (17, 3), mapping indices, default confidence, optional 4th column as confidence.
  - `map_ntu_sequence_to_coco17`: batch mapping (N, 25, 3) → (N, 17, 3).
  - Invalid input shapes and missing columns raise appropriate errors.
- **NTU reader**
  - `sample_id_from_path`: extraction from `.skeleton` and `.skeleton.npy` paths.
  - `load_missing_skeletons_set`, `is_missing_sample`: missing-skeletons list is loaded and used.
  - `read_ntu_npy`, `read_ntu_npy_full`, `read_ntu_skeleton_txt_full`: dict with skel/rgb_xy/depth_xy/tracking_state; invalid/missing keys raise.
  - `read_ntu_sample`: format auto-detection (.npy vs .skeleton).
  - `list_ntu_samples`: discovers files, skips missing-skeleton samples when requested.
  - `select_dominant_body`: most_tracked / closest_z policies.
- **NTU windowing**
  - `slice_windows`: exact/overlapping windows, short-sequence padding (repeat last frame), invalid shape/stride raise.
- **NTU builder**
  - `action_label_from_sample_id`: extraction of A001..A120 from sample ID.
  - `build_dataset`: raises when no NTU samples found under source.
- **Custom10 adapter**
  - Label parsing from folder names (A001_*, A1_*, A43_*), clip discovery (.json, .npy, .skeleton), read_clip and skeleton path (NTU reader + body_to_coco17_normalized), windowing, build_dataset_custom10 with 80/10/10 split; window contract validation for JSON and .skeleton clips.
- **Edge17 adapter** (`test_edge17_reader.py`, `test_edge17_builder.py`)
  - JSONL reader: parse meta + frames, extract keypoints sequence (T, 17, 3), sanitize NaN/Inf.
  - Labels: extract from meta `action_id` or filename, normalize to A### format.
  - Builder: 80/10/10 split by clip_id (no leakage), write parquet + metadata + stats.
- **Feature transforms** (`test_features_transforms.py`)
  - `get_input_features`: raw/norm/vel/angles/combo with conf_mode keep/drop; default config.
  - `build_feature_pipeline`: output shape (T, F) and dtype for raw, norm, vel, angles, combo.
  - `NormalizePoseTransform`: output shape, conf_mode=drop, clamp.
  - `VelocityTransform`: output shape, first-frame zeros.
  - `AnglesTransform`: output shape (T, 10), values in [0, 1].
- **Training metrics** (`test_training_metrics.py`)
  - `accuracy`, `macro_f1`: perfect, partial, zero; labels/num_classes.
  - `confusion_matrix`: shape, diagonal when predictions match.
  - `per_class_precision_recall`: keys (precision, recall, f1, support), list lengths, support values.
- **Training models** (`test_training_models.py`)
  - `get_model`: returns TCN/GRU with correct num_classes and input_features; unknown model raises.
  - TCN/GRU `forward`: output shape (B, num_classes) for default and custom F.
- **Compare runs CLI** (`test_cli_compare_runs.py`)
  - Writes CSV with config + metrics from run dirs; empty runs dir; skips subdirs without checkpoint.

Additional test cases (edge shapes, boundary values, invalid UUIDs, etc.) are in the test modules.

---

## Training Report

For detailed training experiments and results on the Edge17 dataset, see **[training_report.md](training_report.md)**.

### Summary of Best Results (Edge17 Dataset)

| Version | Model | Features | Accuracy | Macro-F1 | Notes |
|---------|-------|----------|----------|----------|-------|
| v1 | TCN | vel | 82.04% | 82.74% | Baseline |
| **v6_lowlr** | TCN | vel | 83.99% | **84.90%** | Best F1 (recommended for deployment) |

**Key findings:**
- Training optimizations (class weights + label smoothing + LR scheduler) improve F1 by +2.16%
- Velocity features are critical for HAR (removing them drops F1 by ~23%)
- TCN outperforms GRU on this dataset

### Exported Model for Cloud Deployment

The best model is exported to `exported_models/edge17_v6_lowlr/`:
- `model.onnx` + `model.onnx.data`: ONNX model
- `model_meta.json`: Input shape, feature spec
- `label_map.json`: Class ID to activity name mapping

**Activity Classes:**
| ID | Activity |
|----|----------|
| 0 | drink water |
| 1 | eat meal |
| 2 | stand up |
| 3 | sit down |
| 4 | reading |
| 5 | falling down |
| 6 | headache |
| 7 | chest pain |
| 8 | back pain |
| 9 | nausea/vomiting |
