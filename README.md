# Object Selection in Intelligent Spaces — Lite

Headless, lightweight pipeline for real-time multi-camera 3D object tracking and pointing-gesture-based object selection, designed to run as a background service in Intelligent Space environments.

> This branch (`Object-Selection-In-Intelligent-Spaces-IS`) contains the **lite version** of the system.
> It removes all visualization and replaces the CLI argument interface with a single JSON configuration file.
> For the full version with visualization and offline video support, see the `main` branch.


## Overview

The system processes synchronized live camera feeds from the IS to:

1. Detect and track objects (people, robots, furniture) using YOLO
2. Match detections across camera views via epipolar geometry
3. Reconstruct 3D positions through RANSAC-robust triangulation
4. Track objects across frames using a 3D SORT (Kalman) tracker
5. Identify the object a person is pointing at using a machine learning gesture classifier
6. Publish results to the IS message broker

The pointing gesture module uses a Logistic Regression classifier trained on 3D skeleton data, combined with a geometric heuristic to determine which arm (left, right, or both) is actively pointing.


## Architecture

```
Camera 0..N  ──────► YOLO Detection ──► Cross-View Matching (Epipolar)
(AMQP / is_wire)                                  │
                                                   ▼
                                        3D Triangulation (RANSAC)
                                                   │
                                                   ▼
                                        3D SORT Tracker (Kalman)
                                                   │
SkeletonsGrouper ─────────────────────────────────►│
(AMQP / is_wire)                                   │
                                                   ▼
                                    Pointing Gesture Classifier
                                    (LogisticRegression + heuristic)
                                                   │
                              ┌────────────────────┤
                              ▼                    ▼
                  is.tracker.detections    MOTPointing.0.Detection
                  (3D positions + bbox)    (pointed object class name)
```


## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `ultralytics` (YOLO), `is-wire`, `is-msgs`, `scikit-learn`, `joblib`, `numpy`, `networkx`, `scipy`, `protobuf`.

> `matplotlib` and `opencv` display functions are **not required** in this version.

YOLO model weights (`.pt`) are **not** included in this repository and must be placed in `models/`.
If you need the models, email gabrielaltoe2017@gmail.com.

### Runtime prerequisites

- **AMQP broker**: set `"address"` in `config_lite.json` with the broker address.
- **Camera topics**: expects `CameraGateway.<cam_id>.Frame` for each camera.
- **Skeleton topic** (optional): expects `SkeletonsGrouper.0.Localization` when `"plot_skeleton": true`.
- **Calibration**: camera calibration files must be in `config_camera/` (e.g. `0.json`, `1.json`, ...).
- **Gesture model**: `models/logisticRegression.sav` required when `"plot_skeleton": true`.


## Running

```bash
python3 -m source.app.main_lite config_lite.json
```


## Data Flow

**Consumes (Subscribe):**

| Topic | Type | Condition |
|-------|------|-----------|
| `CameraGateway.{N}.Frame` | `is_msgs.image_pb2.Image` | Always — one subscription per camera |
| `SkeletonsGrouper.0.Localization` | `is_msgs.image_pb2.ObjectAnnotations` | Only if `"plot_skeleton": true` |

**Publishes:**

| Topic | Type | Condition | Content |
|-------|------|-----------|---------|
| `is.tracker.detections` *(configurable)* | `Detections` (custom proto) | `"publish_detections": true` | Frame number, timestamp, 3D positions and bounding boxes per tracked object |
| `MOTPointing.0.Detection` *(configurable)* | `Struct` | `"publish_pointing": true` + skeleton active | `{ "right_pointing": "ladder", "left_pointing": "None" }` |

The `MOTPointing` message contains the **class name** (e.g. `"ladder"`, `"box"`, `"desk"`) of the object closest to each pointing ray, or `"None"` if no object is in range.


## Configuration (`config_lite.json`)

```jsonc
{
  // Connection
  "address": "10.20.5.2:30000",      // AMQP broker address (host:port)
  "cam_numbers": [0, 1, 2, 3],       // Camera IDs to subscribe to

  // Detection
  "yolo_model": "models/yolo11x.pt", // Path to YOLO weights file
  "confidence": 0.6,                  // YOLO confidence threshold
  "class_list": [0],                  // YOLO class IDs to track (e.g. 0=person)

  // Cross-view matching
  "distance_threshold": 0.4,          // Max epipolar distance for matching
  "drift_threshold": 0.4,             // Drift threshold for matching

  // Triangulation
  "reference_point": "bottom_center", // Reference point on bbox for triangulation
                                      // Options: bottom_center | center | top_center | feet

  // SORT 3D tracker
  "use_3d_tracker": true,             // Enable SORT 3D Kalman tracker
  "max_age": 10,                      // Max frames an object can be missing before removal
  "min_hits": 3,                      // Min detections before a track is confirmed
  "dist_threshold": 1.0,              // Max 3D association distance (meters)

  // IS publishing
  "publish_detections": false,        // Publish 3D detections to AMQP
  "publish_topic": "is.tracker.detections",
  "publish_pointing": true,           // Publish pointing result to AMQP
  "pointing_topic": "MOTPointing.0.Detection",

  // Skeleton / gesture
  "plot_skeleton": true,              // Subscribe to skeleton topic and run gesture classifier

  // Optional output
  "save_coordinates": false,          // Save 3D coordinates to a JSON file
  "output_file": "output.json"
}
```


## Gesture Detection

The pointing gesture classifier (`source/ml/classifier.py`) uses a two-stage pipeline and requires `models/logisticRegression.sav`:

1. **ML gate** — a Logistic Regression model trained on normalized 3D skeleton data classifies whether a pointing gesture is occurring.
2. **Arm selection** — a geometric heuristic determines which arm (left=1, right=2, both=3) is pointing, using:
   - 2D (XY) shoulder–wrist distance (pointing ~0.25 m, resting ~0.11 m)
   - Shoulder–elbow–wrist collinearity angle
   - Wrist elevation relative to shoulder

The selected object is the one whose 3D centroid is **closest along the pointing ray** and within a distance threshold derived from the object's estimated bounding box size.


## Troubleshooting

- If no frames are received, verify the broker address and camera topic names in `config_lite.json`.
- If `MOTPointing` shows `"None"` for all frames:
  - Confirm `SkeletonsGrouper.0.Localization` is publishing.
  - Check if the gesture classifier is triggering (enable `DEBUG` logging to see gesture class output).
  - Verify the YOLO model detects the target objects (check `"class_list"` and `"confidence"`).
- If the SORT tracker loses IDs frequently, reduce `"dist_threshold"` or increase `"max_age"`.
