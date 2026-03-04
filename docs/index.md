# HAR-WindowNet Documentation

HAR-WindowNet is a training workspace for building a data pipeline and machine learning model that uses the same **Window** format as the cloud inference system.

## Goals

1. Use **NTU RGB+D 120** (or custom/Edge17) skeleton and keypoint data.
2. Convert each sample into **Windows** matching the cloud **Contract**.
3. Train a model to classify activity from windowed keypoint sequences.
4. Export a model for inference in the cloud system.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Installation](installation.md) | Environment setup, dependencies, and entry points |
| [Datasets](datasets.md) | Building datasets from NTU, Custom10, and Edge17; validation; source layouts |
| [Window Contract](window-contract.md) | Schema and fields for cloud-compatible windows |
| [Conversion (NTU → Window)](conversion.md) | Projection, confidence, timestamps, mapping, multi-person |
| [Phase A Output](phase-a-output.md) | Directory layout after `build_dataset` |
| [Training & Export](training.md) | Train, evaluate, export to ONNX, inference, compare runs |
| [Troubleshooting](troubleshooting.md) | Common issues and solutions |
| [Testing](testing.md) | Running tests and coverage |
| [Training Report](training-report.md) | Edge17 experiment summary and exported model |

## Quick Start

1. **Install:** See [Installation](installation.md).
2. **Build dataset:** Choose NTU, Custom10, or Edge17 in [Datasets](datasets.md).
3. **Validate:** `python -m har_windownet.cli.validate_dataset --data data_out/<your_dataset>`
4. **Train:** See [Training & Export](training.md).
5. **Export:** ONNX export and inference are documented in [Training & Export](training.md).
