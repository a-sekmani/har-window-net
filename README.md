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

### Validate outputs

```bash
python -m har_windownet.cli.validate_dataset --data data_out/ntu120_windows
```

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

After `build_dataset`, the output directory contains:

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

Phase B consumes Phase A output only (no window building). It adds training (TCN/GRU), evaluation (accuracy, macro-F1, confusion matrix), ONNX export, and offline inference.

**Dependencies**:

- Base: `pip install -e .` (torch, scikit-learn, matplotlib, etc.).
- ONNX export: `pip install -e ".[export]"` (installs `onnx`, `onnxscript`). Export uses **opset 18** to avoid version-conversion failures (requesting 14 caused downgrade errors with current PyTorch/ONNX).
- Inference CLI: requires `onnxruntime`. On **Python 3.14** there may be no wheel — use `pip install onnxruntime` if available, or Python 3.11/3.12 with `pip install -e ".[export,export-inference]"`.

### Train

```bash
python -m har_windownet.cli.train --data data_out/ntu120_windows --model tcn --batch-size 64 --epochs 50 --lr 1e-3 --out runs/exp01
```

- `--data` **must** point to a Phase A dataset directory: it must contain `label_map.json` and `splits/train.parquet` (and val/test). If `label_map.json` is missing, you get a clear error: run `build_dataset` with `--out <dir>` first, then use that path as `--data`.
- Saves `best.ckpt` (by validation macro-F1) and `last.ckpt` under `--out`.

### Eval

```bash
python -m har_windownet.cli.eval --data data_out/ntu120_windows --checkpoint runs/exp01/best.ckpt --split test
```

Writes `reports/test_metrics.json` (accuracy, macro-F1, per-class) and optionally `confusion_matrix.png`.

### Export to ONNX

```bash
python -m har_windownet.cli.export_model --checkpoint runs/exp01/best.ckpt --out runs/exp01/export [--data data_out/ntu120_windows]
```

- Produces `model.onnx`, `model_meta.json`, and `label_map.json` in `--out`. With the new PyTorch ONNX exporter you may see a warning about `dynamic_axes` vs `dynamic_shapes`; export still completes.
- The exported model uses **ONNX opset 18**. Using 14 triggered a failed downgrade (axes adapter) in the ONNX version converter; we use 18 so no conversion is needed and `onnxruntime` can run the model.

### Offline inference on Window JSON

After exporting the model, run inference (requires `onnxruntime`):

```bash
python -m har_windownet.export.inference_onnx --model runs/exp01/export/model.onnx --window data_out/ntu120_windows/samples/window_00000.json
```

Pass a **single-window** JSON file (e.g. one of the files under `samples/`). Returns `pred_label`, `pred_label_id`, and optionally `probs`.

---

## Troubleshooting

| Issue | What to do |
|-------|-------------|
| **No NTU samples found** | `--source` must be a directory containing `*.skeleton` or `*.skeleton.npy`. Use a correct relative path (e.g. `har_windownet/skeleton_files`) or absolute path. |
| **label_map.json not found** (when training) | `--data` must be a Phase A output directory. Run `build_dataset --out <dir>` first, then use that directory as `--data`. |
| **ONNX export: version conversion error** | The project uses **opset 18**; do not lower it to 14 or the ONNX C API converter can fail. Re-export with the default (no change needed in code). |
| **ModuleNotFoundError: onnxruntime** | Install with `pip install onnxruntime` if a wheel exists for your Python version. On Python 3.14 there may be none; use Python 3.11/3.12 for the inference CLI. |
| **ModuleNotFoundError: onnxscript** | Required for `torch.onnx.export`. Install with `pip install -e ".[export]"`. |

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

Additional test cases (edge shapes, boundary values, invalid UUIDs, etc.) are in the test modules.
