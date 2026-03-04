# Training, Evaluation, Export, and Inference

Phase B consumes Phase A output only (no window building). It adds training (TCN/GRU), evaluation (accuracy, macro-F1, confusion matrix, per-class CSV), ONNX export, and offline inference.

Phase C adds a **feature pipeline** (normalization, velocity, angles) and **run configuration**:

- **Feature options** (train/eval/export): `raw` (baseline 51-D), `norm`, `vel`, `angles`, `combo`; `conf_mode`: `keep` (x,y,conf) or `drop` (x,y only); `norm_center` / `norm_scale` (e.g. `auto`, `midhip`, `shoulders`).
- **Run config**: Each training run writes `config.json` in the run directory (data path, model, hyperparameters, `feature_config`). Eval reads `feature_config` from the run dir when present; you must still pass `--data` to the Phase A output directory.
- **Checkpoints** store `input_features` and `feature_config`; export and inference use them so ONNX and offline inference apply the same transforms when needed.
- **Compare runs**: CLI to aggregate all runs under a directory into a single `compare.csv` (run name, model, features, accuracy, macro_f1).

---

## Training steps (end-to-end)

Run these in order from the project root (with the virtual environment activated):

1. **Build dataset** (Phase A). See [Datasets](datasets.md). Example for Custom10:
   ```bash
   python -m har_windownet.cli.build_dataset \
     --dataset custom10 --source datasets --out data_out/custom10 \
     --projection rgb --window-size 30 --stride 30
   ```
   For NTU: `--dataset ntu --source <path_to_ntu> --out data_out/ntu120_windows`.

2. **Validate** (optional): `python -m har_windownet.cli.validate_dataset --data data_out/custom10`

3. **Train**: saves `best.ckpt`, `last.ckpt`, and `config.json` under `--out`. Example:
   ```bash
   python -m har_windownet.cli.train \
     --data data_out/custom10 --model tcn --batch-size 64 --epochs 30 --lr 1e-3 \
     --seed 42 --out runs/custom10_tcn_v1
   ```

4. **Eval**: writes `reports/<split>_metrics.json`, `per_class.csv`, `confusion_matrix.png`, and `class_map.csv`. Pass `--data` to the Phase A output directory (the run’s `config.json` stores this path for reference).
   ```bash
   python -m har_windownet.cli.eval \
     --data data_out/custom10 --checkpoint runs/custom10_tcn_v1/best.ckpt --split test
   ```

5. **Export to ONNX** (optional): uses `input_features` and `feature_config` from the checkpoint; `model_meta.json` includes `feature_spec` when present.
   ```bash
   python -m har_windownet.cli.export_model \
     --checkpoint runs/custom10_tcn_v1/best.ckpt --out runs/custom10_tcn_v1/export \
     --data data_out/custom10
   ```

6. **Inference** (optional): run the exported model on a single-window JSON (or a file containing a list of windows). If the model was trained with a feature pipeline, inference applies the same transforms using `feature_spec` from `model_meta.json`. Optionally: `--export-dir` (directory with `model_meta.json` and `label_map.json`; default: model directory), `--no-probs` (do not output probabilities).
   ```bash
   python -m har_windownet.export.inference_onnx \
     --model runs/custom10_tcn_v1/export/model.onnx \
     --window data_out/custom10/samples/window_00000.json
   ```

7. **Compare runs** (optional): aggregate run configs and test metrics into one CSV.
   ```bash
   python -m har_windownet.cli.compare_runs --runs runs --out runs/compare.csv
   ```

---

## Train

```bash
python -m har_windownet.cli.train --data data_out/ntu120_windows --model tcn --batch-size 64 --epochs 50 --lr 1e-3 --seed 42 --out runs/exp01
```

| Option | Description |
|--------|-------------|
| `--data` | Must point to a Phase A dataset directory (contains `label_map.json` and `splits/train.parquet`). Run `build_dataset` first. |
| `--model` | `tcn` (default) or `gru`. |
| `--seed` | Default 42. Use different seeds (e.g. 42, 7, 123) when reporting mean±std across runs. |
| **Feature options** | `--features`: `raw` (default), `norm`, `vel`, `angles`, `combo`. `--conf-mode`: `keep` or `drop`. `--norm-center`: `auto` (hips), `hips`, `shoulders`. `--norm-scale`: `auto` (shoulders), `shoulders`, `hips`. See `har_windownet.features.transforms`. |
| **Training optimizations** | `--class-weights`, `--label-smoothing <float>`, `--lr-scheduler` (CosineAnnealingLR). |

Saves `best.ckpt` (by validation macro-F1), `last.ckpt`, and **`config.json`** under `--out`. The config holds data path, model name, hyperparameters, and `feature_config` for reproducible eval/export.

---

## Eval

```bash
python -m har_windownet.cli.eval --checkpoint runs/exp01/best.ckpt --split test
```

Or with explicit data: `--data data_out/ntu120_windows --checkpoint runs/exp01/best.ckpt`.

- **Data**: You must pass **`--data`** to the Phase A output directory (e.g. `data_out/ntu120_windows`). The run’s `config.json` stores the data path used at train time for reference.
- **Pipeline**: Eval uses the same feature pipeline as training when `feature_config` is present in the run config or checkpoint.
- Writes under `reports/`: **`<split>_metrics.json`** (accuracy, macro-F1, per-class), **`per_class.csv`**, **`confusion_matrix.png`**, **`class_map.csv`**. Default reports dir is the checkpoint’s parent directory (`run_dir/reports`).

---

## Export to ONNX

```bash
python -m har_windownet.cli.export_model --checkpoint runs/exp01/best.ckpt --out runs/exp01/export [--data data_out/ntu120_windows]
```

- Produces **`model.onnx`**, **`model_meta.json`**, and **`label_map.json`** in `--out`. The checkpoint’s **`input_features`** and **`feature_config`** (if present) are used so the ONNX input shape and metadata match the training pipeline. **`model_meta.json`** includes **`input_features`** and **`feature_spec`** when the model was trained with a feature pipeline. **`--data`** (Phase A root) or **`--label-map`** (path to `label_map.json`) supplies the label map; if neither is set, the default NTU label map is used.
- Exported model uses **ONNX opset 18**. Using 14 can trigger a failed downgrade in the ONNX converter. You may see a PyTorch warning about `dynamic_axes`; export still completes.

---

## Offline inference on Window JSON

After exporting the model, run inference (requires `onnxruntime`):

```bash
python -m har_windownet.export.inference_onnx --model runs/exp01/export/model.onnx --window data_out/ntu120_windows/samples/window_00000.json
```

- **`--model`**: path to `model.onnx`. **`--window`**: path to a single-window JSON file or a JSON file containing a list of windows. **`--export-dir`**: directory containing `model_meta.json` and `label_map.json` (default: directory of the model file). **`--no-probs`**: omit probability output.
- Returns `pred_label`, `pred_label_id`, and optionally `probs` per window.
- If **`model_meta.json`** contains **`feature_spec`**, the inference code applies the same feature transforms (normalize, velocity, angles, etc.) to the window keypoints before running the model, so raw Window JSON (30×17×3) works for both baseline and feature-pipeline models.

---

## Compare runs

Aggregate run directories (each with optional `config.json` and `reports/test_metrics.json`) into a single CSV:

```bash
python -m har_windownet.cli.compare_runs --runs runs --out runs/compare.csv
```

- **`--runs`** (default: `runs`): directory containing one subdirectory per run. Only subdirs that have **`best.ckpt`** or **`last.ckpt`** are included.
- **`--out`**: output CSV path (default: `<runs>/compare.csv`).
- CSV columns: **run**, **model**, **features**, **conf_mode**, **accuracy**, **macro_f1**. Metrics come from `reports/test_metrics.json` when present; config from `config.json`. Old runs without config/metrics still get a row with default model/features and empty accuracy/macro_f1.

---

## Run directory layout (after train + eval)

A typical run directory (e.g. `runs/exp01`) contains:

- **`config.json`**: data path, model, batch_size, epochs, lr, seed, device, **feature_config** (written at train start).
- **`best.ckpt`**, **`last.ckpt`**: PyTorch checkpoints (model_state_dict, epoch, val_macro_f1, num_classes, model_name, **input_features**, **feature_config**).
- **`reports/`** (after eval): **`<split>_metrics.json`**, **`per_class.csv`**, **`confusion_matrix.png`**, **`class_map.csv`**.
