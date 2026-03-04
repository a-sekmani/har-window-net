# Testing

Tests are in the `tests/` directory and use **pytest**.

## Run all tests

With the virtual environment activated and dev dependencies installed (`pip install -e ".[dev]"`):

```bash
pytest
```

From the project root, this discovers and runs all tests under `tests/`.

## Run with verbose output

```bash
pytest -v
```

## Run a specific test file or test

```bash
pytest tests/test_contracts_window.py -v
pytest tests/test_contracts_window.py::test_window_contract_valid -v
```

## Run tests by keyword

```bash
pytest -k "window" -v
pytest -k "mapping" -v
```

## Test coverage (optional)

If you install `pytest-cov`:

```bash
pip install pytest-cov
pytest --cov=har_windownet --cov-report=term-missing
```

## What is tested

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
