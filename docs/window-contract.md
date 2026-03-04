# Window Contract

A window represents one person (track_id), one camera, and a fixed-length segment of N frames. The schema is compatible with the cloud inference system.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| id | UUID str | Unique window id |
| device_id | str | e.g. `ntu-offline` |
| camera_id | str | e.g. `ntu-cam` |
| session_id | UUID str | Per video/sample |
| track_id | int | Usually 1 |
| ts_start_ms, ts_end_ms | int | Start time (e.g. 0 offline); end = **floor**((window_size−1)×(1000/fps)) + ts_start_ms — see [Conversion](conversion.md#timestamps-offline-ntu) |
| fps | float | 30.0 (float for cloud compatibility) |
| window_size | int | 30 |
| mean_pose_conf | float | Mean keypoint confidence (from tracking_state or default) |
| label | str | Activity label (e.g. from NTU) |
| label_source | `"dataset"` | Fixed |
| created_at | ISO str | Build timestamp |
| keypoints | array | Shape [T][K][3] = [30][17][3], (x, y, conf); x,y in [0..1] |
| source_body_id | int? | NTU body index used (0 = dominant); for multi-person traceability |

- **T = 30** frames, **K = 17** keypoints (COCO-17), each (x, y, confidence).
- Implementation and validation: `har_windownet.contracts.window` (WindowContract, validate_window_dict).
