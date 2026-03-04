# Troubleshooting

| Issue | What to do |
|-------|-------------|
| **No NTU samples found** | `--source` must be a directory containing `*.skeleton` or `*.skeleton.npy`. Use a correct relative path (e.g. `har_windownet/skeleton_files`) or absolute path. |
| **label_map.json not found** (when training) | `--data` must be a Phase A output directory. Run `build_dataset --out <dir>` first, then use that directory as `--data`. |
| **ONNX export: version conversion error** | The project uses **opset 18**; do not lower it to 14 or the ONNX C API converter can fail. Re-export with the default (no change needed in code). |
| **ModuleNotFoundError: onnxruntime** | Install with `pip install onnxruntime` if a wheel exists for your Python version. On Python 3.14 there may be none; use Python 3.11/3.12 for the inference CLI. |
| **ModuleNotFoundError: onnxscript** | Required for `torch.onnx.export`. Install with `pip install -e ".[export]"`. |
| **Eval: Missing --data** | The eval CLI always requires `--data` pointing to the Phase A output directory (e.g. `data_out/ntu120_windows`). Use the same path as in the run’s `config.json` if re-evaluating. |
