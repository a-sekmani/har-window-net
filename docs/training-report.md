# Training Report

For detailed training experiments and results on the Edge17 dataset, see **[training_report.md](../training_report.md)** (in the project root).

## Summary of best results (Edge17 dataset)

| Version | Model | Features | Accuracy | Macro-F1 | Notes |
|---------|-------|----------|----------|----------|-------|
| v1 | TCN | vel | 82.04% | 82.74% | Baseline |
| **v6_lowlr** | TCN | vel | 83.99% | **84.90%** | Best F1 (recommended for deployment) |

**Findings:**

- Training optimizations (class weights + label smoothing + LR scheduler) improve F1 by +2.16%.
- Velocity features are critical for HAR (removing them drops F1 by ~23%).
- TCN outperforms GRU on this dataset.

## Exported model for cloud deployment

The best model is exported to `exported_models/edge17_v6_lowlr/`:

- `model.onnx`: ONNX model (single file)
- `model_meta.json`: Input shape, feature spec
- `label_map.json`: Class ID to activity name mapping

**Activity classes** (class ID matches `label_to_id`: A001→0, A002→1, A008→2, A009→3, …):

| ID | Label | Activity |
|----|-------|----------|
| 0 | A001 | drink water |
| 1 | A002 | eat meal |
| 2 | A008 | sitting down |
| 3 | A009 | standing up |
| 4 | A011 | reading |
| 5 | A043 | falling down |
| 6 | A044 | headache |
| 7 | A045 | chest pain |
| 8 | A046 | back pain |
| 9 | A048 | nausea/vomiting |
