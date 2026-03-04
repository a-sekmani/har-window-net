# Phase A output layout

After `build_dataset` (NTU, Custom10, or Edge17), the output directory has the same structure.

## Directory structure

```
data_out/ntu120_windows/
  label_map.json          # id_to_name, label_to_id (A001..A120), num_classes
  dataset_meta.json       # source_dir, projection, window_size, stride, fps, seed, counts
  splits/
    train.parquet
    val.parquet
    test.parquet
  stats/                  # always written
    class_counts.json     # window count per label (e.g. "A001": 123)
    pose_conf_hist.json   # mean_pose_conf: bin_edges, counts (10 bins), min, max, mean
  samples/                # only if --export-samples N > 0
    window_00000.json
    window_00001.json
    ...
```

- Parquet files are standard Arrow/Parquet (signature PAR1). Each row is one window with columns matching the [Window Contract](window-contract.md); `keypoints` is stored as nested list `[window_size][17][3]` (default `window_size` is 30, from `dataset_meta.json`).
- Use this directory as `--data` for training and (when needed) for eval/export.
