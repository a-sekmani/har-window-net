# Conversion (NTU → Window)

This document describes the choices made when converting NTU skeleton data into cloud-compatible windows.

## Projection (3D → 2D)

NTU 3D (x,y,z) is in camera space, not pixels. Normalizing by “scene range” is **unstable** across samples (camera distance varies). Prefer:

- **RGB projection** when available: use `colorX/colorY` (`.skeleton`) or `rgb_body0` (`.npy`), then normalize by **1920×1080**.
- **Depth projection** otherwise: use `depthX/depthY`, then normalize by **512×424**.
- **3D** only as fallback: use (x,y) from 3D and normalize by scene bounds (documented as unstable).

The pipeline supports an explicit **`--projection rgb|depth|3d`** option (default: **rgb** when available). See `har_windownet/datasets/ntu/config.py` and `preprocess.py`.

## Confidence

Do not use a fixed `1.0`. Use Kinect V2 **trackingState** per joint (from `.skeleton`):

- **2 (Tracked)** → conf = 1.0  
- **1 (Inferred)** → conf = 0.5  
- **0 (NotTracked)** → conf = 0.0  

`mean_pose_conf` is computed from these values. For `.npy` (no tracking state), default 1.0 is used. See `har_windownet/datasets/ntu/mapping.tracking_state_to_confidence`.

## Timestamps (offline NTU)

For offline NTU, timestamps are synthetic but **consistent**. Use **floor** (integer truncation) so training and cloud agree; do **not** mix with `round()`.

- `ts_start_ms`: 0 for the first window, then `stride × (1000/fps)` per subsequent window.
- `ts_end_ms = ts_start_ms + floor((window_size - 1) * (1000 / fps))`  
  Example: window_size=30, fps=30, ts_start_ms=0 → 29×(1000/30)=966.66… → **966** (not 967).

Helper: `har_windownet.contracts.window.ts_end_ms_from_window(window_size, fps, ts_start_ms)`.

## Mapping (25 → 17)

COCO-17 order is fixed; same order in training and cloud. Nose/ears/eyes have no direct NTU joint and use **Head(3)** as proxy; their confidence can be 0 if Head is not tracked (some actions depend on head). See `har_windownet/datasets/ntu/mapping.py`.

## Multi-person

NTU can have more than one body per frame. Policy is **fixed**:

- **Dominant body**: choose by **most tracked joints** (`tracking_state == 2`), or by **closest z** (mean spine-base z).  
  Use `har_windownet.datasets.ntu.reader.select_dominant_body(bodies, policy="most_tracked"|"closest_z")`.
- Store **`source_body_id`** (body index used) in each window and/or in `dataset_meta.json` for traceability.
