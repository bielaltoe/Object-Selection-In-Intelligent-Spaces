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
python3 -m source.app.main \
  --realtime \
  --cam_numbers 0 1 2 3 \
  --plot_skeleton \
  --yolo_model models/yolo11x.pt
```

Offline (from recorded videos):

```bash
python3 -m source.app.main \
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

The pointing gesture classifier (`source/ml/classifier.py`) uses a two-stage pipeline and requires `models/logisticRegression.sav` to run:

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
