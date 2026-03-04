# Installation

## Environment Setup

From the project root, create a virtual environment, activate it, and install the project with dev tools (for tests):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows:

```bash
.venv\Scripts\activate
```

To install without dev tools (no pytest/ruff):

```bash
pip install -e .
```

Verify installation:

```bash
python -c "import har_windownet; print(har_windownet.__version__)"
```

## Dependencies

- **Base:** `pip install -e .` installs PyTorch, scikit-learn, matplotlib, and other runtime dependencies.
- **ONNX export:** `pip install -e ".[export]"` adds `onnx` and `onnxscript`. Export uses **opset 18** for compatibility.
- **Inference CLI:** requires `onnxruntime`. On Python 3.14 a wheel may not be available; use Python 3.11 or 3.12 with `pip install -e ".[export,export-inference]"` if needed.

## Entry Points

After `pip install -e .`, these entry points are available:

| Command | Description |
|--------|-------------|
| `har-windownet-build-dataset` | Build dataset from NTU, Custom10, or Edge17 |
| `har-windownet-validate-dataset` | Validate Phase A output |
| `har-windownet-train` | Train TCN or GRU model |
| `har-windownet-eval` | Evaluate checkpoint on a split |
| `har-windownet-export-model` | Export checkpoint to ONNX |
| `har-windownet-compare-runs` | Aggregate run configs and metrics to CSV |

You can also run any CLI as a module:

```bash
python -m har_windownet.cli.build_dataset ...
python -m har_windownet.cli.validate_dataset ...
python -m har_windownet.cli.train ...
python -m har_windownet.cli.eval ...
python -m har_windownet.cli.export_model ...
python -m har_windownet.cli.compare_runs ...
```
