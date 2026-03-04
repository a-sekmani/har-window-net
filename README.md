# HAR-WindowNet

A training workspace for building a data pipeline and ML model that uses the same **Window** format as the cloud system.

## Goals

1. Use **NTU RGB+D 120** (or custom/Edge17) skeleton and keypoint data.
2. Convert each sample into **Windows** matching the cloud **Contract**.
3. Train a model to classify activity from windowed keypoint sequences.
4. Export a model for inference in the cloud system.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m har_windownet.cli.build_dataset --source <path_to_ntu> --out data_out/ntu120_windows
python -m har_windownet.cli.validate_dataset --data data_out/ntu120_windows
python -m har_windownet.cli.train --data data_out/ntu120_windows --model tcn --out runs/exp01
```

For Custom10 or Edge17, see [docs/datasets.md](docs/datasets.md). Full workflow: [docs/training.md](docs/training.md).

## Documentation

| Topic | Document |
|-------|----------|
| **Installation** | [docs/installation.md](docs/installation.md) — Environment, dependencies, entry points |
| **Datasets** | [docs/datasets.md](docs/datasets.md) — Build from NTU, Custom10, Edge17; validate; source layouts |
| **Window contract** | [docs/window-contract.md](docs/window-contract.md) — Schema and fields |
| **Conversion (NTU → Window)** | [docs/conversion.md](docs/conversion.md) — Projection, confidence, timestamps, mapping |
| **Phase A output** | [docs/phase-a-output.md](docs/phase-a-output.md) — Directory layout after `build_dataset` |
| **Training & export** | [docs/training.md](docs/training.md) — Train, eval, ONNX export, inference, compare runs |
| **Troubleshooting** | [docs/troubleshooting.md](docs/troubleshooting.md) — Common issues and fixes |
| **Testing** | [docs/testing.md](docs/testing.md) — pytest usage and coverage |
| **Training report** | [docs/training-report.md](docs/training-report.md) — Edge17 summary; full report: [training_report.md](training_report.md) |

Full index: [docs/index.md](docs/index.md).

## Training report (Edge17)

Detailed experiments and results: **[training_report.md](training_report.md)**. Best model (v6_lowlr, TCN, vel): **84.90%** macro-F1; exported to `exported_models/edge17_v6_lowlr/`.
