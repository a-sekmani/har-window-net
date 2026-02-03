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

- `--source`: Path to the NTU folder (files `.skeleton` or `raw_npy/` with `.npy` from [NTU_RGBD120_Parser](https://github.com/FesianXu/NTU_RGBD120_Parser_python)).
- `--out`: Output directory (e.g. `data_out/ntu120_windows`).
- `--projection`: **rgb** (default when available), **depth**, or **3d**. Use rgb/depth for stable 2D normalization; 3d is unstable across samples.
- `--export-samples`: Number of windows to export under `samples/` (optional).

### Validate outputs

```bash
python -m har_windownet.cli.validate_dataset --data data_out/ntu120_windows
```

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
| ts_start_ms, ts_end_ms | int | Start time (e.g. 0 offline); end = (window_size-1)*(1000/fps) |
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

For offline NTU, timestamps are synthetic but **consistent**:

- `ts_start_ms = 0` (or a fixed base)
- `ts_end_ms = (window_size - 1) * (1000 / fps)`  
  Helper: `har_windownet.contracts.window.ts_end_ms_from_window(window_size, fps, ts_start_ms)`

### Mapping (25 → 17)

COCO-17 order is fixed; same order in training and cloud. Nose/ears/eyes have no direct NTU joint and use **Head(3)** as proxy; their confidence can be 0 if Head is not tracked (some actions depend on head). See `har_windownet/datasets/ntu/mapping.py`.

### Multi-person

NTU can have more than one body per frame. Policy is **fixed**:

- **Dominant body**: choose by **most tracked joints** (`tracking_state == 2`), or by **closest z** (mean spine-base z).  
  Use `har_windownet.datasets.ntu.reader.select_dominant_body(bodies, policy="most_tracked"|"closest_z")`.
- Store **`source_body_id`** (body index used) in each window and/or in `dataset_meta.json` for traceability.

---

## Phase A output layout

```
data_out/ntu120_windows/
  label_map.json
  dataset_meta.json
  splits/
    train.parquet
    val.parquet
    test.parquet
  samples/
    sample_100_windows.json   # if requested
  stats/
    class_counts.json
    pose_conf_hist.json
```

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
  - `read_ntu_npy`: dict with `skel_body0`; invalid/missing keys raise.
  - `read_ntu_sample`: format auto-detection (.npy vs .skeleton).
  - `list_ntu_samples`: discovers files, skips missing-skeleton samples when requested.

Additional test cases (edge shapes, boundary values, invalid UUIDs, etc.) are added where applicable and documented in the test modules.
