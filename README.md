# Object Selection in Intelligent Spaces

Real-time multi-camera system for 3D object tracking and pointing-gesture-based object selection, designed for Human-Robot Interaction (HRI) in Intelligent Space environments.


## Overview

The system processes synchronized video feeds from multiple cameras to:

1. Detect and track objects (people, robots, furniture) using YOLO
2. Match detections across camera views via epipolar geometry
3. Reconstruct 3D positions through RANSAC-robust triangulation
4. Identify the object a person is pointing at using a machine learning gesture classifier

The pointing gesture module uses a Logistic Regression classifier trained on 3D skeleton data, combined with a geometric heuristic to determine which arm (left, right, or both) is actively pointing.

## Architecture

```
Camera 0..N (AMQP / is_wire)
        │
        ▼
  YOLO Detection  ──►  Cross-View Matching (Epipolar)
                                │
                                ▼
                        3D Triangulation (RANSAC)
                                │
                                ▼
                        3D SORT Tracker (Kalman)
                                │
SkeletonsGrouper ──────────────►│
(3D human skeleton)             │
                                ▼
                    Pointing Gesture Classifier
                    (LogisticRegression + heuristic)
                                │
                                ▼
                    Selected Object  ──►  ROS2 (optional)
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `opencv-python`, `ultralytics` (YOLO), `is-wire`, `is-msgs`, `scikit-learn`, `joblib`, `numpy`, `matplotlib`, `networkx`, `pandas`, `scipy`, `protobuf`, `tabulate`.

Optional dependencies provided by a ROS 2 installation (not via pip): `rclpy`, `rosbag2_py`, `geometry_msgs`, `nav_msgs`, `std_msgs`, `irobot_create_msgs`.

YOLO model weights (`.pt`) are **not** included in this repository and must be downloaded separately and placed in `models/`.
If you need the models, email gabrielaltoe2017@gmail.com.

### Runtime prerequisites

- **AMQP broker**: update `source/protobuf/config.json` with the broker address.
- **Camera topics**: live mode expects `CameraGateway.<cam_id>.Frame`.
- **Skeleton topic (optional)**: `--plot_skeleton` expects `SkeletonsGrouper.0.Localization` to be available.
- **Calibration**: camera calibration files live in `config_camera/` (e.g. `0.json`, `1.json`, ...).

## Usage

```bash
python source/main.py \
  --realtime \
  --cam_numbers 0 1 2 3 \
  --plot_skeleton \
  --yolo_model models/yolo11x.pt
```

Offline (from recorded videos):

```bash
python source/main.py \
  --video_path experiment_sas/experiments1/videos \
  --cam_numbers 0 1 2 3 \
  --yolo_model models/yolo11x.pt
```

### Key flags

| Flag | Description |
|------|-------------|
| `--cam_numbers` | Camera IDs to use (e.g. `0 1 2 3`) |
| `--realtime` | Use live AMQP camera streams (default is offline mode) |
| `--video_path` | Directory containing recorded videos (offline mode) |
| `--yolo_model` | Path to YOLO `.pt` weights file |
| `--plot_skeleton` | Enable 3D skeleton and pointing ray visualization |
| `--experiment_log` | Log pointing data to CSV for analysis |
| `--send_to_ros` | Send selected object footprints to ROS 2 |
| `--publish` | Publish 3D coordinates to an AMQP topic |

## Gesture Detection

The pointing gesture classifier (`source/classifier.py`) uses a two-stage pipeline and requires `models/logisticRegression.sav` to run:

1. **ML gate** — a Logistic Regression model (`models/logisticRegression.sav`) trained on normalized 3D skeleton data classifies whether a pointing gesture is occurring.
2. **Arm selection** — a geometric heuristic on the normalized skeleton determines which arm (left=1, right=2, both=3) is pointing, using:
   - 2D (XY) shoulder–wrist distance (pointing ~0.25 m, resting ~0.11 m)
   - Shoulder–elbow–wrist collinearity angle
   - Wrist elevation relative to shoulder (pointing arm stays near shoulder height, resting arm hangs ~0.49 m below)

## Outputs

- **3D coordinates (JSON)**: `--save_coordinates --output_file output.json`
- **Experiment log (CSV)**: `--experiment_log --experiment_output experiment_results.csv`
- **Benchmark log (CSV)**: `--benchmark --benchmark_output benchmark_results.csv`
- **Video output**: `--save-video --output-video output.mp4`
- **Exported figures**: `--export_figures --figures_output_dir exported_figures`
- **AMQP publish**: `--publish --publish_topic is.tracker.detections`

## Troubleshooting

- Use `--headless` on machines without a display server.
- If no frames are received in real-time mode, verify the broker address and camera topics.
- If `--plot_skeleton` shows no gesture lines, confirm `SkeletonsGrouper.0.Localization` is publishing.

---

## Lite Mode (`main_lite.py`)

A lightweight, **headless** version of the pipeline designed to run as a background service in the Intelligent Space.
It removes all visualization (matplotlib, video mosaic, 3D plot, correspondence graph) and replaces the `argparse` CLI with a single JSON config file.

### When to use

| | `main.py` | `main_lite.py` |
|---|---|---|
| Visualization | ✅ | ❌ |
| Offline video files | ✅ | ❌ |
| Live IS cameras | ✅ | ✅ |
| JSON config | ❌ | ✅ |
| Low resource footprint | ❌ | ✅ |

### Running

```bash
python3 -m source.app.main_lite config_lite.json
```

### Data flow

**Consumes (Subscribe):**

| Topic | Type | Condition |
|-------|------|-----------|
| `CameraGateway.{N}.Frame` | `is_msgs.image_pb2.Image` | Always (one per camera) |
| `SkeletonsGrouper.0.Localization` | `is_msgs.image_pb2.ObjectAnnotations` | Only if `plot_skeleton: true` |

**Publishes:**

| Topic | Type | Condition | Content |
|-------|------|-----------|---------|
| `is.tracker.detections` *(configurable)* | `Detections` (custom proto) | `publish_detections: true` | 3D positions + bounding boxes per tracked object |
| `MOTPointing.0.Detection` *(configurable)* | `Struct` (JSON-like) | `publish_pointing: true` + skeleton active | `{ "right_pointing": "ladder", "left_pointing": "None" }` |

### Configuration (`config_lite.json`)

```jsonc
{
  "address": "10.20.5.2:30000",     // AMQP broker address
  "cam_numbers": [0, 1, 2, 3],      // Camera IDs to subscribe to

  "yolo_model": "models/yolo11x.pt", // Path to YOLO weights
  "confidence": 0.6,                 // YOLO detection threshold
  "class_list": [0],                 // YOLO class IDs to track (0 = person)

  "distance_threshold": 0.4,         // Max distance for cross-view matching
  "drift_threshold": 0.4,            // Drift threshold for matching
  "reference_point": "bottom_center",// Triangulation reference point on bbox
                                     // Options: bottom_center | center | top_center | feet

  "max_age": 10,                     // SORT: max frames object can be missing
  "min_hits": 3,                     // SORT: min hits to start tracking
  "dist_threshold": 1.0,             // SORT: max 3D distance for association (meters)
  "use_3d_tracker": true,            // Enable SORT 3D tracker

  "publish_detections": false,       // Publish 3D detections to AMQP
  "publish_topic": "is.tracker.detections",
  "publish_pointing": true,          // Publish pointing result to AMQP
  "pointing_topic": "MOTPointing.0.Detection",

  "plot_skeleton": true,             // Subscribe to skeleton topic and run gesture classifier

  "save_coordinates": false,         // Save 3D coordinates to JSON file
  "output_file": "output.json"
}
```
