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

- `model.onnx` + `model.onnx.data`: ONNX model
- `model_meta.json`: Input shape, feature spec
- `label_map.json`: Class ID to activity name mapping

**Activity classes:**

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
