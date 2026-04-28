"""
Multi-View Person Tracking and 3D Reconstruction System

This script implements a real-time multi-camera tracking system that:
1. Processes video feeds from 4 cameras simultaneously
2. Detects and tracks people using YOLO
3. Matches detections across different views
4. Performs 3D triangulation
5. Visualizes results in real-time

Dependencies:
- OpenCV, NumPy, Matplotlib, NetworkX
- Custom modules: ploting_utils, video_loader, tracker, triangulation, matcher
"""

import argparse
import csv
import json
import logging
import math
import os
import socket
import time

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, patheffects
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add path for gesture detection module
import networkx as nx
from source.config.config import CLASS_NAMES
from google.protobuf.json_format import Parse
from google.protobuf.message import Message as PbMessage
from source.visualization.graph_visualization import visualize_graph

# Import the newly created modules
from source.io.io_utils import save_3d_coordinates_with_ids

# Intelligent space imports
from is_msgs.image_pb2 import Image
from is_wire.core import Channel, Message, Subscription
from source.io.live_video_loader import (
    StreamChannel,
    load_json,
    publish,
    publish_with_3d_bbox,
    to_np,
)
from source.core.matcher import Matcher
from source.visualization.ploting_utils import Utils
from source.core.three_dimentional_tracker import SORT_3D
from source.core.tracker import Tracker
from source.core.triangulation import triangulate_ransac
from source.io.video_loader import VideoLoader
from source.visualization.visualization_utils import draw_bbox, visualize_camera_positions

from source.ml.classifier import Gesture
from is_msgs.image_pb2 import HumanKeypoints as HKP
from is_msgs.image_pb2 import ObjectAnnotations
from source.messaging.publish_to_ros import (
    send_object_footprint_to_ros,
    should_publish,
    update_publish_cache,
)

# Skeleton keypoint mapping (COCO format)
TO_COCO_IDX = {
    HKP.Value("NOSE"): 0,
    HKP.Value("LEFT_EYE"): 1,
    HKP.Value("RIGHT_EYE"): 2,
    HKP.Value("LEFT_EAR"): 3,
    HKP.Value("RIGHT_EAR"): 4,
    HKP.Value("LEFT_SHOULDER"): 5,
    HKP.Value("RIGHT_SHOULDER"): 6,
    HKP.Value("LEFT_ELBOW"): 7,
    HKP.Value("RIGHT_ELBOW"): 8,
    HKP.Value("LEFT_WRIST"): 9,
    HKP.Value("RIGHT_WRIST"): 10,
    HKP.Value("LEFT_HIP"): 11,
    HKP.Value("RIGHT_HIP"): 12,
    HKP.Value("LEFT_KNEE"): 13,
    HKP.Value("RIGHT_KNEE"): 14,
    HKP.Value("LEFT_ANKLE"): 15,
    HKP.Value("RIGHT_ANKLE"): 16,
}

# Skeleton links for visualization
SKELETON_LINKS = [
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("RIGHT_SHOULDER")),
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_HIP")),
    (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_HIP")),
    (HKP.Value("LEFT_HIP"), HKP.Value("RIGHT_HIP")),
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_EAR")),
    (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_EAR")),
    (HKP.Value("LEFT_SHOULDER"), HKP.Value("LEFT_ELBOW")),
    (HKP.Value("LEFT_ELBOW"), HKP.Value("LEFT_WRIST")),
    (HKP.Value("LEFT_HIP"), HKP.Value("LEFT_KNEE")),
    (HKP.Value("LEFT_KNEE"), HKP.Value("LEFT_ANKLE")),
    (HKP.Value("RIGHT_SHOULDER"), HKP.Value("RIGHT_ELBOW")),
    (HKP.Value("RIGHT_ELBOW"), HKP.Value("RIGHT_WRIST")),
    (HKP.Value("RIGHT_HIP"), HKP.Value("RIGHT_KNEE")),
    (HKP.Value("RIGHT_KNEE"), HKP.Value("RIGHT_ANKLE")),
    (HKP.Value("NOSE"), HKP.Value("LEFT_EYE")),
    (HKP.Value("LEFT_EYE"), HKP.Value("LEFT_EAR")),
    (HKP.Value("NOSE"), HKP.Value("RIGHT_EYE")),
    (HKP.Value("RIGHT_EYE"), HKP.Value("RIGHT_EAR")),
]


import numpy as np


def distancia_ponto_para_reta_3d(C, P0, P1):
    """
    Computes the distance from point C to a semi-ray that starts at P0 and
    points toward P1.

    C = point (object centroid)
    P0 = ray origin (pointing origin, e.g., wrist)
    P1 = point along the pointing direction

    Returns:
        - Perpendicular distance if the point is in front of the ray
        - np.inf if the point is behind the ray
    """
    v = P1 - P0  # ray direction vector
    w = C - P0   # vector from origin to the point

    # Scalar projection of w on v (normalized by |v|)
    # t > 0 means the point is along the pointing direction
    # t <= 0 means the point is behind or at the ray origin
    v_norm_sq = np.dot(v, v)
    if v_norm_sq < 1e-9:  # avoid division by zero when P0 == P1
        return np.linalg.norm(w)
    
    t = np.dot(w, v) / v_norm_sq
    
    # If t <= 0, the point is behind the ray origin
    if t <= 0:
        return np.inf
    
    # Perpendicular distance when the point is in front of the ray
    cross = np.cross(w, v)
    dist = np.linalg.norm(cross) / np.linalg.norm(v)

    return dist


def generate_prism(bl, br, tr, tl, min_size=0.3):
    """
    Generate a 3D prism from the 4 corners of a quadrilateral.

    Args:
        bl, br, tr, tl: cantos bottom-left, bottom-right, top-right, top-left
        min_size: minimum prism size (depth)

    Returns:
        faces: list of prism faces
        height: prism height/depth
    """
    bl = np.array(bl)
    br = np.array(br)
    tr = np.array(tr)
    tl = np.array(tl)

    # --- Face normal ---
    v1 = br - bl
    v2 = tl - bl
    normal = np.cross(v1, v2)
    norm_length = np.linalg.norm(normal)
    if norm_length < 1e-9:
        # If the normal is too small, use a default direction
        normal = np.array([0, 0, 1])
    else:
        normal = normal / norm_length

    # --- Use face size as depth, with a minimum ---
    height = max(np.linalg.norm(br - bl), min_size)

    # --- Offset for the opposite face ---
    shift = normal * height

    # Face oposta
    bl_b = bl + shift
    br_b = br + shift
    tr_b = tr + shift
    tl_b = tl + shift

    # --- Prism faces ---
    faces = []

    # Front
    faces.append([bl, br, tr, tl])

    # Back
    faces.append([bl_b, br_b, tr_b, tl_b])

    # Sides
    faces.append([bl, tl, tl_b, bl_b])
    faces.append([br, tr, tr_b, br_b])

    # Top and bottom
    faces.append([tl, tr, tr_b, tl_b])
    faces.append([bl, br, br_b, bl_b])

    return faces, height


def generate_prism_from_centroid(centroid, width, depth, height, min_size=0.2):
    """
    Generate a 3D prism centered at the centroid with specified dimensions.
    More stable than using the 4 triangulated corners directly.

    Args:
        centroid: prism center (x, y, z)
        width: prism width (X axis)
        depth: prism depth (Y axis)
        height: prism height (Z axis)
        min_size: minimum size for each dimension

    Returns:
        faces: list of prism faces
        diagonal: prism diagonal (used as distance threshold)
    """
    centroid = np.array(centroid)

    # Enforce minimum sizes
    width = max(width, min_size)
    depth = max(depth, min_size)
    height = max(height, min_size)

    # Compute the 8 vertices of the centered prism
    hw, hd, hh = width / 2, depth / 2, height / 2

    # Base vertices (z = centroid[2] - hh)
    z_bottom = centroid[2] - hh
    z_top = centroid[2] + hh

    # Lower base (4 corners)
    v0 = np.array([centroid[0] - hw, centroid[1] - hd, z_bottom])  # front-left
    v1 = np.array([centroid[0] + hw, centroid[1] - hd, z_bottom])  # front-right
    v2 = np.array([centroid[0] + hw, centroid[1] + hd, z_bottom])  # back-right
    v3 = np.array([centroid[0] - hw, centroid[1] + hd, z_bottom])  # back-left
    
    # Upper base (4 corners)
    v4 = np.array([centroid[0] - hw, centroid[1] - hd, z_top])  # front-left
    v5 = np.array([centroid[0] + hw, centroid[1] - hd, z_top])  # front-right
    v6 = np.array([centroid[0] + hw, centroid[1] + hd, z_top])  # back-right
    v7 = np.array([centroid[0] - hw, centroid[1] + hd, z_top])  # back-left
    
    # --- Prism faces ---
    faces = []
    
    # Lower base
    faces.append([v0, v1, v2, v3])
    
    # Upper base
    faces.append([v4, v5, v6, v7])
    
    # Side faces
    faces.append([v0, v1, v5, v4])  # frente
    faces.append([v2, v3, v7, v6])  # back
    faces.append([v0, v3, v7, v4])  # esquerda
    faces.append([v1, v2, v6, v5])  # direita
    
    # Prism diagonal (used as distance threshold)
    diagonal = np.sqrt(width**2 + depth**2 + height**2) / 2
    
    return faces, diagonal


# Global Gesture object for skeleton line plotting
obj = Gesture()

# Set global font size to 12 (same as plot_from_json.py)
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["lines.linewidth"] = 1.0

# Set up matplotlib style for white background
# plt.style.use('default')
PLOT_COLORS = list(plt.cm.tab10.colors) + list(plt.cm.Set1.colors)


def plt_reta(sk: np.ndarray, ax, y: int = 0, option: bool = False):
    # reta_para_plot returns 10 values; we only need the first 6 for plotting lines here.
    rightx, righty, rightz, leftx, lefty, leftz, *_ = obj.reta_para_plot(sk)

    (
        y,
        left_distancia,
        left_teta,
        right_distancia,
        right_teta,
    ) = obj.classificador_ml(sk)

    if y == 3 or y == 0:
        ax.plot(
            rightx,
            righty,
            rightz,
            color="g",
        )

        ax.plot(
            leftx,
            lefty,
            leftz,
            color="b",
        )

    elif y == 2:
        ax.plot(
            rightx,
            righty,
            rightz,
            color="g",
        )

    elif y == 1:
        ax.plot(
            leftx,
            lefty,
            leftz,
            color="b",
        )

last_sended_coordinates = []
COORD_CHANCHE_THRESHOLD = 0.1


def main():
    """
    Main execution function that orchestrates the multi-camera tracking pipeline.

    Pipeline steps:
    1. Load video streams
    2. Initialize tracking and matching components
    3. Process each frame:
       - Detect and track objects
       - Match detections across views
       - Perform 3D triangulation
       - Visualize results
    4. Save output video
    """
    bbheight = 0.0
    argparser = argparse.ArgumentParser(
        description="Multi-Camera Tracking and 3D Reconstruction"
    )
    argparser.add_argument(
        "--video_path", type=str, default=".", help="Path to video files"
    )
    argparser.add_argument(
        "--output_file",
        type=str,
        default="output.json",
        help="Output JSON file for 3D coordinates",
    )
    argparser.add_argument(
        "--save_coordinates",
        default=False,
        action="store_true",
        help="Save 3D coordinates to JSON file",
    )
    argparser.add_argument(
        "--use_3d_tracker",
        default=True,
        action="store_true",
        help="Use algorithm for 3D tracking",
    )
    argparser.add_argument(
        "--max_age",
        type=int,
        default=10,
        help="Maximum frames object can be missing (SORT)",
    )
    argparser.add_argument(
        "--min_hits",
        type=int,
        default=3,
        help="Minimum hits to start tracking (SORT)",
    )
    argparser.add_argument(
        "--dist_threshold",
        type=float,
        default=1.0,
        help="Maximum distance for association (SORT)",
    )
    argparser.add_argument(
        "--class_list",
        type=int,
        nargs="+",
        default=[0],
        help="List of classes to track (e.g., 0 1 2) (default: 0 for person)",
    )
    argparser.add_argument(
        "--yolo_model",
        type=str,
        default="models/yolo11x.pt",
        help="Path to YOLO model file",
    )
    argparser.add_argument(
        "--confidence",
        type=float,
        default=0.6,
        help="Confidence threshold for YOLO detection",
    )
    argparser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.4,
        help="Distance threshold for matching",
    )
    argparser.add_argument(
        "--drift_threshold",
        type=float,
        default=0.4,
        help="Drift threshold for matching",
    )
    argparser.add_argument(
        "--reference_point",
        type=str,
        default="bottom_center",
        choices=["bottom_center", "center", "top_center", "feet"],
        help="Reference point on bounding box for triangulation",
    )

    # Plot-related arguments (all enabled by default, disable with flags)
    argparser.add_argument(
        "--headless",
        action="store_true",
        help="Disable all visualization (headless mode)",
    )
    argparser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable correspondence graph visualization",
    )
    argparser.add_argument(
        "--no-3d", action="store_true", help="Disable 3D plot visualization"
    )
    argparser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video mosaic visualization",
    )
    argparser.add_argument(
        "--save-video",
        action="store_true",
        default=False,
        help="Save output video regardless of visualization settings",
    )
    argparser.add_argument(
        "--output-video",
        type=str,
        default="output.mp4",
        help="Path to save the output video",
    )
    argparser.add_argument(
        "--publish",
        action="store_true",
        default=False,
        help="Publish 3D coordinates to channel",
    )
    argparser.add_argument(
        "--publish_topic",
        type=str,
        default="is.tracker.detections",
        help="Topic to publish 3D coordinates to",
    )
    argparser.add_argument(
        "--cam_numbers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="List of camera numbers to process (default: [0, 1, 2, 3])",
    )
    argparser.add_argument(
        "--realtime",
        action="store_true",
        default=False,
        help="Run in real-time mode (process frames from a realtime camera feed)",
    )
    argparser.add_argument(
        "--plot_skeleton",
        action="store_true",
        default=False,
        help="Enable 3D skeleton and pointing line visualization (requires SkeletonsGrouper topic)",
    )
    # Arguments for exporting figures
    argparser.add_argument(
        "--export_figures",
        action="store_true",
        help="Enable exporting of final plots (video mosaic, graph, 3D plot).",
    )
    argparser.add_argument(
        "--export_dpi", type=int, default=300, help="DPI for exported figures."
    )
    argparser.add_argument(
        "--figures_output_dir",
        type=str,
        default="exported_figures",
        help="Directory to save exported figures.",
    )
    argparser.add_argument(
        "--send_to_ros",
        action="store_true",
        default=False,
        help="Send 3D footprint coordinates to ROS",
    )
    argparser.add_argument(
        "--experiment_log",
        action="store_true",
        default=False,
        help="Enable experiment data collection for pointing gesture analysis",
    )
    argparser.add_argument(
        "--experiment_output",
        type=str,
        default="experiment_results.csv",
        help="Output CSV file for experiment data (default: experiment_results.csv)",
    )
    argparser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Enable performance benchmarking (frame rate, pipeline latency, per-stage timing)",
    )
    argparser.add_argument(
        "--benchmark_output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file for benchmark data (default: benchmark_results.csv)",
    )

    args = argparser.parse_args()

    # Get arguments from parser
    video_path = args.video_path
    output_json_file = args.output_file
    save_flag = args.save_coordinates
    use_sort = args.use_3d_tracker
    max_age = args.max_age
    min_hits = args.min_hits
    dist_threshold = args.dist_threshold
    class_list = args.class_list
    yolo_model = args.yolo_model
    confidence = args.confidence
    distance_threshold = args.distance_threshold
    drift_threshold = args.drift_threshold
    reference_point = args.reference_point
    output_video_path = args.output_video
    publish_topic = args.publish_topic

    # Handle visualization settings with opt-out approach
    show_plot = not args.headless
    show_graph = show_plot and not args.no_graph
    show_3d = show_plot and not args.no_3d
    show_video = show_plot and not args.no_video
    save_video = args.save_video
    publish_flag = args.publish
    send_to_ros_flag = args.send_to_ros  # Always send to ROS for footprint tracking
    cam_numbers = args.cam_numbers
    realtime_flag = args.realtime
    plot_skeleton = args.plot_skeleton

    # Arguments for exporting figures
    export_figures = args.export_figures
    export_dpi = args.export_dpi
    figures_output_dir = args.figures_output_dir
    dist_centroid_left = 0.0
    dist_centroid_right = 0.0
    
    # Experiment data collection
    experiment_log_enabled = args.experiment_log
    experiment_output_file = args.experiment_output
    experiment_log = []  # List to store experiment data entries
    
    # Performance benchmarking
    benchmark_enabled = args.benchmark
    benchmark_output_file = args.benchmark_output
    benchmark_log = []  # List to store per-frame benchmark entries

    if benchmark_enabled:
        logging.info(f"Performance benchmarking enabled. Output file: {benchmark_output_file}")

    if experiment_log_enabled:
        logging.info(f"Experiment data collection enabled. Output file: {experiment_output_file}")
    if publish_flag:
        logging.info(f"Publishing to topic: {publish_topic}")

    if not realtime_flag:
        logging.info(f"Video path: {video_path}")
    logging.info(f"Output file: {output_json_file}")
    logging.info(f"Using SORT 3D tracker: {use_sort}")
    logging.info(
        f"Visualization: Plot={show_plot}, Graph={show_graph}, 3D={show_3d}, Video={show_video}"
    )
    logging.info(f"Reference point for triangulation: {reference_point}")
    if plot_skeleton:
        logging.info(
            "Skeleton plotting enabled - will subscribe to SkeletonsGrouper topic"
        )

    if not os.path.exists(video_path):
        raise ValueError(f"Video path not found: {video_path}")

    if not realtime_flag:
        video_files = os.listdir(video_path)
        logging.info(f"Video files: {video_files}")

        video_files = [
            os.path.join(video_path, f).replace("\\", "/") for f in video_files
        ]
        logging.info(f"Video files: {video_files}")

    # Ensure cam_numbers is a list of integers
    if isinstance(cam_numbers, int):
        cam_numbers = [cam_numbers]
    elif isinstance(cam_numbers, str):
        cam_numbers = [int(num) for num in cam_numbers.split(",")]
    # Ensure cam_numbers are unique and sorted
    cam_numbers = sorted(set(cam_numbers))

    logging.info(f"Cam numbers: {cam_numbers}")

    # Create utils instance
    utils = Utils()
    tracker = Tracker(
        [yolo_model for _ in range(len(cam_numbers))],
        cam_numbers,
        class_list,
        confidence,
    )
    matcher = Matcher(distance_threshold, drift_threshold)

    # Initialize SORT 3D tracker if requested
    if use_sort:
        sort_tracker = SORT_3D(
            max_age=max_age, min_hits=min_hits, dist_threshold=dist_threshold
        )
        logging.info("SORT 3D tracker initialized")

    # Store fixed prism dimensions per track_id
    # Once estimated, dimensions remain fixed for that object
    prism_dimensions_cache = {}

    # Set up matplotlib figure with interactive backend and white background only if plotting is enabled
    video_ax = graph_ax = ax_3d = video_img = fig = None

    if show_plot:
        plt.ion()

        # Create figure with white background
        fig = plt.figure(figsize=(20, 12), facecolor="white")

        # Create a dynamic layout based on which plots are enabled
        if show_graph and show_3d and show_video:
            # Create a 2x2 layout with all plots
            gs = GridSpec(
                2,
                2,
                figure=fig,
                width_ratios=[1.3, 1],
                wspace=0.05,
                hspace=0.1,
            )

            # Video mosaic (left column spanning two rows)
            video_ax = fig.add_subplot(
                gs[:, 0]
            )  # Spans both rows in first column

            # Graph visualization (top-right)
            graph_ax = fig.add_subplot(gs[0, 1])

            # 3D plot (bottom-right)
            ax_3d = fig.add_subplot(
                gs[1, 1], projection="3d", facecolor="white"
            )
        elif show_3d and show_video:
            # Create a 1x2 layout without graph
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.05)

            # Video mosaic (left column)
            video_ax = fig.add_subplot(gs[0, 0])

            # 3D plot (right column)
            ax_3d = fig.add_subplot(
                gs[0, 1], projection="3d", facecolor="white"
            )
        elif show_graph and show_video:
            # Create a 1x2 layout without 3D
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.05)

            # Video mosaic (left column)
            video_ax = fig.add_subplot(gs[0, 0])

            # Graph visualization (right column)
            graph_ax = fig.add_subplot(gs[0, 1])
        elif show_graph and show_3d:
            # Create a 1x2 layout with graph and 3D plots (no video)
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.05)

            # Graph visualization (left column)
            graph_ax = fig.add_subplot(gs[0, 0])

            # 3D plot (right column)
            ax_3d = fig.add_subplot(
                gs[0, 1], projection="3d", facecolor="white"
            )
        elif show_video:
            # Only video
            video_ax = fig.add_subplot(111)
        elif show_3d:
            # Only 3D plot
            ax_3d = fig.add_subplot(111, projection="3d", facecolor="white")
        elif show_graph:
            # Only graph
            graph_ax = fig.add_subplot(111)

        # Configure axes if they exist
        if video_ax:
            video_ax.axis("off")
            video_img = video_ax.imshow(
                np.zeros((720, 1280, 3), dtype=np.uint8)
            )
            video_title = video_ax.set_title(
                "Multi-Camera View",
                fontsize=16,
                color="black",
                fontweight="bold",
                pad=15,
            )

        if ax_3d:
            # Configure 3D plot with white background styling - ONCE at initialization
            def configure_3d_axis(ax):
                """Configure 3D axis properties that need to be preserved"""
                ax.set_xlim([-4, 4])
                ax.set_ylim([-4, 4])
                ax.set_zlim([0, 4])

                # Set prominent axis labels that are clearly visible
                ax.set_xlabel("X (m)", fontsize=12, labelpad=13, color="black")
                ax.set_ylabel("Y (m)", fontsize=12, labelpad=13, color="black")
                ax.set_zlabel("Z (m)", fontsize=12, labelpad=13, color="black")

                # Set grid and tick colors for white background
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor("lightgray")
                ax.yaxis.pane.set_edgecolor("lightgray")
                ax.zaxis.pane.set_edgecolor("lightgray")
                ax.tick_params(axis="x", colors="black", labelsize=10)
                ax.tick_params(axis="y", colors="black", labelsize=10)
                ax.tick_params(axis="z", colors="black", labelsize=10)

                ax.xaxis._axinfo["grid"]["color"] = "lightgray"
                ax.yaxis._axinfo["grid"]["color"] = "lightgray"
                ax.zaxis._axinfo["grid"]["color"] = "lightgray"

            # Apply initial configuration
            configure_3d_axis(ax_3d)

            # Store the configuration function for reuse after clear()
            ax_3d._configure_func = configure_3d_axis

        # # Add a title to the figure if any plotting is enabled
        if show_plot:
            if reference_point == "center":
                ref_desc = "Center of bbox"
            elif reference_point == "top_center":
                ref_desc = "Top-center of bbox"
            elif reference_point == "feet":
                ref_desc = "20% above bottom-center (feet)"
            else:  # bottom_center
                ref_desc = "Bottom-center of bbox"

            fig.suptitle(
                f"Reference point: {ref_desc}",
                fontsize=15,
                color="black",
                fontweight="bold",
                y=0.98,
            )

    pos = {}  # Position cache for graph nodes

    # Initialize video writer if saving video
    video_writer = None

    frame_number = 0  # Reset frame number for each iteration
    # Add variables to track camera feed health

    last_frame_time = {}
    connections_healthy = True
    
    # FPS estimation for video recording
    frame_timestamps = []  # Store timestamps of successfully processed frames
    estimated_fps = 10.0  # Default FPS, will be updated dynamically
    fps_window_size = 30  # Number of frames to average for FPS estimation

    # Create dictionaries to store channels and subscriptions for each camera
    channels = {}
    subscriptions = {}
    camera_status = {}  # Track camera health
    with open("source/protobuf/config.json", "r") as f:
        config = json.load(f)
    publish_channel = StreamChannel(f"amqp://guest:guest@{config['address']}")

    # Create a channel and subscription for each camera
    for cam_idx in cam_numbers:
        try:
            channels[cam_idx] = StreamChannel(
                f"amqp://guest:guest@{config['address']}"
            )
            subscriptions[cam_idx] = Subscription(
                channels[cam_idx], name=f"CameraCapture{cam_idx}"
            )
            subscriptions[cam_idx].subscribe(
                topic=f"CameraGateway.{cam_idx}.Frame"
            )
            camera_status[cam_idx] = {
                "connected": True,
                "last_seen": time.time(),
            }
            logging.info(f"Successfully subscribed to camera {cam_idx}")
        except Exception as e:
            logging.error(f"Failed to subscribe to camera {cam_idx}: {e}")
            camera_status[cam_idx] = {"connected": False, "last_seen": None}

    # Create skeleton channel and subscription if skeleton plotting is enabled
    skeleton_channel = None
    skeleton_subscription = None
    skeleton_data = None  # Will store the latest skeleton data
    if plot_skeleton:
        try:
            skeleton_channel = StreamChannel(
                f"amqp://guest:guest@{config['address']}"
            )
            skeleton_subscription = Subscription(
                skeleton_channel, name="SkeletonSubscriber"
            )
            skeleton_subscription.subscribe(
                topic="SkeletonsGrouper.0.Localization"
            )
            logging.info(
                "Successfully subscribed to SkeletonsGrouper.0.Localization for skeleton plotting"
            )
        except Exception as e:
            logging.error(f"Failed to subscribe to skeleton topic: {e}")
            plot_skeleton = (
                False  # Disable skeleton plotting if subscription fails
            )

    while True:
        frames = []
        current_time = time.time()
        frame_collected = [False] * len(cam_numbers)

        # Collect frames from all cameras with timeout
        t_frame_start = time.perf_counter()
        start_collection_time = time.time()
        collection_timeout = 1.0  # 1 second timeout for frame collection

        while (
            not all(frame_collected)
            and (time.time() - start_collection_time) < collection_timeout
        ):
            for cam_idx, cam in enumerate(cam_numbers):
                if frame_collected[cam_idx]:
                    continue  # Skip if frame already collected for this camera

                try:
                    message, dropped = channels[cam].consume_last()
                    if message is not None:
                        image = message.unpack(Image)
                        frame = to_np(image)

                        if frame is not None:
                            # Ensure frames list has enough slots
                            while len(frames) <= cam_idx:
                                frames.append(None)

                            frames[cam_idx] = frame
                            frame_collected[cam_idx] = True
                            last_frame_time[cam] = current_time

                            if dropped > 10:
                                logging.warning(
                                    f"Camera {cam}: Dropped {dropped} frames"
                                )

                except socket.timeout:
                    # Expected timeout when no new frames available
                    continue
                except Exception as e:
                    logging.error(f"Error reading from camera {cam}: {e}")
                    # Mark as failed for this iteration
                    frame_collected[cam_idx] = False

        # Check if we have frames from all cameras
        valid_frames = sum(frame_collected)
        if valid_frames == len(cam_numbers):
            logging.info(
                f"Frame {frame_number}: Successfully collected frames from all {len(cam_numbers)} cameras"
            )

            # Consume skeleton data if skeleton plotting is enabled
            skeleton_pts = None
            if plot_skeleton and skeleton_channel is not None:
                try:
                    skeleton_msg, _ = skeleton_channel.consume_last()
                    if skeleton_msg is not None:
                        skeleton_results = skeleton_msg.unpack(
                            ObjectAnnotations
                        )
                        n_persons = len(skeleton_results.objects)

                        if n_persons >= 1:
                            pts = np.zeros(
                                (n_persons, 17, 3), dtype=np.float64
                            )
                            for i, skeleton in enumerate(
                                skeleton_results.objects
                            ):
                                for part in skeleton.keypoints:
                                    if part.id in TO_COCO_IDX:
                                        pts[i, TO_COCO_IDX[part.id], 0] = (
                                            part.position.x
                                        )
                                        pts[i, TO_COCO_IDX[part.id], 1] = (
                                            part.position.y
                                        )
                                        pts[i, TO_COCO_IDX[part.id], 2] = (
                                            part.position.z
                                        )

                            # If multiple persons, select the one closest to origin
                            if False:
                                pts_mean = np.mean(pts, axis=1)
                                origin = np.zeros((n_persons, 3))
                                distance = np.linalg.norm(
                                    pts_mean - origin, axis=1
                                )
                                idx = distance.argmin()
                                skeleton_pts = pts[idx, :, :].reshape(1, 17, 3)
                            else:
                                skeleton_pts = pts

                            logging.debug(
                                f"Skeleton data received: {n_persons} person(s)"
                            )
                except socket.timeout:
                    pass  # No new skeleton data available
                except Exception as e:
                    logging.warning(f"Error consuming skeleton data: {e}")

            # Process the complete frame set
            t_collection_end = time.perf_counter()

            graph = nx.Graph()
            t_detection_start = time.perf_counter()
            tracker.detect_and_track(frames)
            detections = tracker.get_detections()
            t_detection_end = time.perf_counter()
            triangulated_points = []
            graph_component_ids = []
            node_color_map = {}

            # Process detections and build graph
            for d in detections:
                id = int(d.id)
                bbox = d.bbox
                cam = int(d.cam)
                frame = d.frame
                centroid = d.centroid
                name = d.name
                graph.add_node(
                    f"cam{cam}id{id}",
                    bbox=bbox,
                    id=id,
                    frame=frame,
                    centroid=centroid,
                    name=name,
                )

                # Get color based on ID for consistency
                color_rgb = utils.id_to_rgb_color(id)

                # Get class name string if available
                class_name = CLASS_NAMES.get(int(name), f"Class {int(name)}")

                # Draw bounding box
                frames[cam] = draw_bbox(
                    frames[cam],
                    bbox,
                    class_name,
                    id,
                    color_rgb,
                    reference_point,
                )

            # Match detections and build edges
            t_matching_start = time.perf_counter()
            for k in cam_numbers:
                for j in cam_numbers:
                    if k != j and k < j:
                        matches = matcher.match_detections(detections, [k, j])
                        for match in matches:
                            n1 = f"cam{k}id{int(match[0].id)}"
                            n2 = f"cam{j}id{int(match[1].id)}"
                            if n1 in graph.nodes and n2 in graph.nodes:
                                graph.add_edge(n1, n2)

            # Process triangulation
            t_matching_end = time.perf_counter()
            t_triangulation_start = time.perf_counter()
            class_ids = []
            track_colors = {}
            bbox_3d_list = []  # Store 3D bounding box info for each detection

            for idx, c in enumerate(nx.connected_components(graph)):
                subgraph = graph.subgraph(c)
                if len(subgraph.nodes) > 1:
                    ids = sorted(subgraph.nodes)
                    d2_points = []
                    proj_matricies = []
                    bboxes_points = []
                    class_id = subgraph.nodes[ids[0]]["name"]

                    # Collect 2D bbox corners for all cameras viewing this object
                    bbox_corners_2d = {
                        "top_left": [],
                        "top_right": [],
                        "bottom_left": [],
                        "bottom_right": [],
                        "bottom_center": [],
                        "top_center": [],
                        "centroid": [],
                    }

                    for node in ids:
                        cam = int(node.split("cam")[1].split("id")[0])
                        id = int(node.split("id")[1])
                        centroid = subgraph.nodes[node]["centroid"]
                        bbox = subgraph.nodes[node]["bbox"]
                        P_cam = matcher.P_all[cam]

                        # Get reference point based on user selection
                        if reference_point == "center":
                            point_2d = (
                                (bbox[2] + bbox[0]) / 2,
                                (bbox[3] + bbox[1]) / 2,
                            )
                        elif reference_point == "top_center":
                            point_2d = ((bbox[2] + bbox[0]) / 2, bbox[1])
                        elif reference_point == "feet":
                            bottom_offset = 0.2 * (bbox[3] - bbox[1])
                            point_2d = (
                                (bbox[2] + bbox[0]) / 2,
                                bbox[3] - bottom_offset,
                            )
                        else:  # bottom_center
                            point_2d = ((bbox[2] + bbox[0]) / 2, bbox[3])

                        # Extract all 4 corners of the bbox for 3D cube triangulation
                        # bbox format: [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right
                        bbox_corners_2d["top_left"].append(
                            ((bbox[0], bbox[1]), P_cam)
                        )
                        bbox_corners_2d["top_right"].append(
                            ((bbox[2], bbox[1]), P_cam)
                        )
                        bbox_corners_2d["bottom_left"].append(
                            ((bbox[0], bbox[3]), P_cam)
                        )
                        bbox_corners_2d["bottom_right"].append(
                            ((bbox[2], bbox[3]), P_cam)
                        )
                        bbox_corners_2d["bottom_center"].append(
                            (((bbox[0] + bbox[2]) / 2, bbox[3]), P_cam)
                        )
                        bbox_corners_2d["top_center"].append(
                            (((bbox[0] + bbox[2]) / 2, bbox[1]), P_cam)
                        )
                        
                        bbox_corners_2d["centroid"].append((centroid, P_cam))

                        d2_points.append(point_2d)
                        bboxes_points.append(bbox)
                        proj_matricies.append(P_cam)

                    if len(d2_points) >= 2:
                        # Triangulate the main reference point
                        point_3d, _ = triangulate_ransac(
                            proj_matricies, d2_points
                        )

                        # Triangulate all 4 bbox corners to create a 3D footprint/cube
                        bbox_3d_corners = {}
                        for (
                            corner_name,
                            corner_data,
                        ) in bbox_corners_2d.items():
                            if len(corner_data) >= 2:
                                corner_points = [c[0] for c in corner_data]
                                corner_proj_matrices = [
                                    c[1] for c in corner_data
                                ]
                                corner_3d, _ = triangulate_ransac(
                                    corner_proj_matrices, corner_points
                                )
                                bbox_3d_corners[corner_name] = corner_3d

                        # Store bbox_3d_corners with the detection for later use
                        # Create 3D bounding box: use bottom corners as base, estimate height from top corners
                        bbox_3d_info = None
                        if (
                            "bottom_center" in bbox_3d_corners
                            and "top_center" in bbox_3d_corners
                        ):
                            height_3d = (
                                bbox_3d_corners["top_center"][2]
                                - bbox_3d_corners["bottom_center"][2]
                            )
                            # Store as: [center_x, center_y, center_z, width, depth, height]
                            # Width and depth estimated from corner distances
                            if (
                                "bottom_left" in bbox_3d_corners
                                and "bottom_right" in bbox_3d_corners
                            ):
                                width_3d = np.linalg.norm(
                                    np.array(
                                        bbox_3d_corners["bottom_right"][:2]
                                    )
                                    - np.array(
                                        bbox_3d_corners["bottom_left"][:2]
                                    )
                                )
                            else:
                                width_3d = 0.5  # default width

                            # Store the 3D bbox info (can be used for publishing)
                            bbox_3d_info = {
                                "corners": bbox_3d_corners,
                                "center": point_3d,
                                "width": width_3d,
                                "height": (
                                    abs(height_3d) if height_3d else 1.7
                                ),  # default human height
                            }
                            # Attach to subgraph for later access
                            subgraph.graph["bbox_3d"] = bbox_3d_info

                        triangulated_points.append(point_3d)
                        bbox_3d_list.append(
                            bbox_3d_info
                        )  # Store bbox info for this detection
                        graph_component_ids.append(idx)
                        class_ids.append(class_id)

                        temp_color_rgb = utils.id_to_rgb_color(idx)
                        track_colors[idx] = temp_color_rgb

                        for node in subgraph.nodes:
                            node_color_map[node] = utils.normalize_rgb_color(
                                temp_color_rgb
                            )

            # SORT tracking and visualization code remains the same...
            # ...existing code for SORT tracking...
            t_triangulation_end = time.perf_counter()

            # If we're using SORT, update the 3D tracker
            t_tracking_start = time.perf_counter()
            if use_sort and triangulated_points:
                sort_result = sort_tracker.update(
                    triangulated_points, class_ids
                )
                point_3d_list = sort_result["positions"]
                track_ids = sort_result["ids"]
                sorted_class_ids = sort_result["class_ids"]
                trajectories = sort_result["trajectories"]

                # Update colors based on actual track IDs from SORT
                track_colors.clear()
                node_color_map.clear()

                for i, (point_3d, track_id) in enumerate(
                    zip(point_3d_list, track_ids)
                ):
                    color_rgb = utils.id_to_rgb_color(track_id)
                    track_colors[track_id] = color_rgb

                    if i < len(triangulated_points):
                        for comp_idx, orig_point in enumerate(
                            triangulated_points
                        ):
                            distance = np.linalg.norm(
                                np.array(point_3d) - np.array(orig_point)
                            )
                            if distance < 0.1:
                                components = list(
                                    nx.connected_components(graph)
                                )
                                if comp_idx < len(components):
                                    component_nodes = components[comp_idx]
                                    for node in component_nodes:
                                        node_color_map[node] = (
                                            utils.normalize_rgb_color(
                                                color_rgb
                                            )
                                        )
                                break

                logging.info(
                    f"Frame {frame_number}: SORT tracking {len(point_3d_list)} objects"
                )
            else:
                point_3d_list = triangulated_points
                track_ids = graph_component_ids
                trajectories = {}
                sorted_class_ids = class_ids

            t_tracking_end = time.perf_counter()

            # Save coordinates if requested
            t_publish_start = time.perf_counter()
            if save_flag and point_3d_list:
                if use_sort:
                    save_3d_coordinates_with_ids(
                        frame_number,
                        point_3d_list,
                        track_ids,
                        output_json_file,
                        sorted_class_ids,
                    )
                else:
                    save_3d_coordinates_with_ids(
                        frame_number,
                        point_3d_list,
                        track_ids,
                        output_json_file,
                        class_ids,
                    )

            if publish_flag and point_3d_list:
                # Use publish_with_3d_bbox to send both center points and 3D bounding boxes
                publish_with_3d_bbox(
                    channel=publish_channel,
                    frame=frame_number,
                    point_3d_list=point_3d_list,
                    track_ids=track_ids,
                    class_ids=sorted_class_ids,
                    bbox_3d_list=bbox_3d_list,
                    topic=publish_topic,
                )

            # Create video mosaic with annotations and better formatting if video visualization is enabled
            t_publish_end = time.perf_counter()
            t_visualization_start = time.perf_counter()

            if show_plot and show_video:
                processed_frames = []
                for idx, frame in enumerate(frames):
                    # Convert BGR to RGB for matplotlib
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Add a nice camera label
                    h, w = rgb_frame.shape[:2]
                    overlay = rgb_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (175, 40), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, rgb_frame, 0.3, 0, rgb_frame)
                    cv2.putText(
                        rgb_frame,
                        f"Camera {idx}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    processed_frames.append(cv2.resize(rgb_frame, (540, 360)))

                if processed_frames:
                    # Arrange frames in a grid that matches the number of active cameras
                    border_px = 5
                    frame_h, frame_w = processed_frames[0].shape[:2]
                    num_frames = len(processed_frames)
                    cols = math.ceil(math.sqrt(num_frames))
                    rows = math.ceil(num_frames / cols)
                    mosaic_h = rows * frame_h + (rows - 1) * border_px
                    mosaic_w = cols * frame_w + (cols - 1) * border_px
                    full_mosaic = (
                        np.ones((mosaic_h, mosaic_w, 3), dtype=np.uint8) * 255
                    )

                    for idx, processed in enumerate(processed_frames):
                        row = idx // cols
                        col = idx % cols
                        y = row * (frame_h + border_px)
                        x = col * (frame_w + border_px)
                        full_mosaic[y : y + frame_h, x : x + frame_w] = (
                            processed
                        )
                else:
                    full_mosaic = np.zeros((720, 1280, 3), dtype=np.uint8)

                # Add frame counter to the video mosaic
                cv2.rectangle(
                    full_mosaic,
                    (full_mosaic.shape[1] - 200, full_mosaic.shape[0] - 50),
                    (full_mosaic.shape[1], full_mosaic.shape[0]),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    full_mosaic,
                    f"Frame: {frame_number}",
                    (full_mosaic.shape[1] - 190, full_mosaic.shape[0] - 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                video_img.set_data(full_mosaic)
                video_title.set_text(
                    f"Multi-Camera View - {len(point_3d_list)} Objects Detected"
                )

            # Update 3D plot with better styling
            if show_plot and show_3d:
                ax_3d.clear()

                # Efficiently restore axis configuration using stored function
                if hasattr(ax_3d, "_configure_func"):
                    ax_3d._configure_func(ax_3d)
                else:
                    # Fallback: simple axis labels only
                    ax_3d.set_xlabel(
                        "X (m)", fontsize=11, labelpad=12, color="black"
                    )
                    ax_3d.set_ylabel(
                        "Y (m)", fontsize=11, labelpad=12, color="black"
                    )
                    ax_3d.set_zlabel(
                        "Z (m)", fontsize=11, labelpad=12, color="black"
                    )
                    ax_3d.tick_params(axis="x", colors="black", labelsize=10)
                    ax_3d.tick_params(axis="y", colors="black", labelsize=10)
                    ax_3d.tick_params(axis="z", colors="black", labelsize=10)

                # Add ground plane and enhanced camera positions for better spatial understanding
                if hasattr(matcher, "P_all"):
                    try:
                        visualize_camera_positions(ax_3d, matcher.P_all)
                    except Exception as e:
                        logging.warning(
                            f"Error visualizing camera positions: {str(e)}"
                        )

                # Define class-specific markers (same as plot_from_json.py)
                class_markers = {
                    0: "o",  # pessoa (person)
                    1: "o",  # capacete (helmet)
                    3: "^",  # robo (robot)
                    56: "s",  # cadeira (chair)
                }
                # # List of available markers: https://matplotlib.org/stable/api/markers_api.html
                # all_markers = {
                #     'point': '.',
                #     'pixel': ',',
                #     'circle': 'o',
                #     'triangle_down': 'v',
                #     'triangle_up': '^',
                #     'triangle_left': '<',
                #     'triangle_right': '>',
                #     'tri_down': '1',
                #     'tri_up': '2',
                #     'tri_left': '3',
                #     'tri_right': '4',
                #     'square': 's',
                #     'pentagon': 'p',
                #     'star': '*',
                #     'hexagon1': 'h',
                #     'hexagon2': 'H',
                #     'plus': '+',
                #     'x': 'x',
                #     'd': 'd',
                #     'thin_diamond': 'D',
                #     'vline': '|',
                #     'hline': '_',
                #     'None': 'None',
                #     'tickleft': 't',
                #     'tickright': 'T',
                #     'tickup': 'u',
                #     'tickdown': 'd',
                #     'caretleft': '<',
                #     'caretright': '>',
                #     'caretup': '^',
                #     'caretdown': 'v',
                #     'caretleftbase': 'l',
                #     'caretrightbase': 'r',
                #     'caretupbase': 'u',
                #     'caretdownbase': 'd',
                #     'dash': '-',
                #     'solid_line': '_',
                #     'custom': 'None'
                # }
                # Define class names for legend (same as plot_from_json.py)
                # Collect legend information

                legend_elements = []

                # Pre-compute pointing gesture data before drawing objects
                # This allows us to determine which objects are being pointed at
                pointing_data = None
                if plot_skeleton and skeleton_pts is not None:
                    for person_idx in range(skeleton_pts.shape[0]):
                        pts = skeleton_pts[person_idx]

                        # Draw skeleton links
                        for link in SKELETON_LINKS:
                            begin, end = link
                            if begin in TO_COCO_IDX and end in TO_COCO_IDX:
                                begin_idx = TO_COCO_IDX[begin]
                                end_idx = TO_COCO_IDX[end]

                                x_pair = [pts[begin_idx][0], pts[end_idx][0]]
                                y_pair = [pts[begin_idx][1], pts[end_idx][1]]
                                z_pair = [pts[begin_idx][2], pts[end_idx][2]]

                                # Skip if any coordinate is zero (missing keypoint)
                                if not (
                                    np.all(pts[begin_idx] == 0)
                                    or np.all(pts[end_idx] == 0)
                                ):
                                    ax_3d.plot(
                                        x_pair,
                                        y_pair,
                                        z_pair,
                                        linewidth=3,
                                        color="black",
                                        alpha=0.8,
                                        zorder=6,
                                    )

                        # Compute pointing lines using the Gesture classifier
                        try:
                            # Normalize skeleton for classification
                            skM = obj.normalizacao(pts)
                            df = obj.list_to_dataframe(skM)

                            # Use classifier to determine pointing gesture
                            (
                                gesture_class,
                                left_distancia,
                                left_teta,
                                right_distancia,
                                right_teta,
                            ) = obj.classificador_ml(pts)

                            # Get pointing line coordinates
                            (
                                rightx,
                                righty,
                                rightz,
                                leftx,
                                lefty,
                                leftz,
                                left_end,
                                right_end,
                            ) = obj.reta_para_plot(pts, length=5)

                            right_start = np.array(
                                [rightx[0], righty[0], rightz[0]]
                            )
                            left_start = np.array(
                                [leftx[0], lefty[0], leftz[0]]
                            )

                            # Store pointing data for use when drawing objects
                            pointing_data = {
                                "gesture_class": gesture_class,
                                "right_start": right_start,
                                "right_end": right_end,
                                "left_start": left_start,
                                "left_end": left_end,
                                "rightx": rightx,
                                "righty": righty,
                                "rightz": rightz,
                                "leftx": leftx,
                                "lefty": lefty,
                                "leftz": leftz,
                            }

                            # Draw pointing lines based on gesture classification
                            # gesture_class: 0=none, 1=left pointing, 2=right pointing, 3=both pointing
                            if gesture_class == 3:
                                # Draw both lines (green for right, blue for left)
                                ax_3d.plot(
                                    rightx,
                                    righty,
                                    rightz,
                                    color="g",
                                    linewidth=2,
                                    alpha=0.9,
                                    zorder=7,
                                    label="Right pointing",
                                )
                                ax_3d.plot(
                                    leftx,
                                    lefty,
                                    leftz,
                                    color="b",
                                    linewidth=2,
                                    alpha=0.9,
                                    zorder=7,
                                    label="Left pointing",
                                )
                                ax_3d.scatter(
                                    right_end[0],
                                    right_end[1],
                                    right_end[2],
                                    s=20,
                                    color="red",
                                    zorder=10,
                                )
                                ax_3d.scatter(
                                    left_end[0],
                                    left_end[1],
                                    left_end[2],
                                    s=20,
                                    color="red",
                                    zorder=10,
                                )
                            elif gesture_class == 2:
                                # Draw only right pointing line (green)
                                ax_3d.plot(
                                    rightx,
                                    righty,
                                    rightz,
                                    color="g",
                                    linewidth=2,
                                    alpha=0.9,
                                    zorder=7,
                                    label="Right pointing",
                                )
                                ax_3d.scatter(
                                    right_end[0],
                                    right_end[1],
                                    right_end[2],
                                    s=20,
                                    color="red",
                                    zorder=10,
                                )
                            elif gesture_class == 1:
                                # Draw only left pointing line (blue)
                                ax_3d.plot(
                                    leftx,
                                    lefty,
                                    leftz,
                                    color="b",
                                    linewidth=2,
                                    alpha=0.9,
                                    zorder=7,
                                    label="Left pointing",
                                )
                                ax_3d.scatter(
                                    left_end[0],
                                    left_end[1],
                                    left_end[2],
                                    s=20,
                                    color="red",
                                    zorder=10,
                                )

                        except Exception as e:
                            logging.debug(f"Error processing pointing gesture: {e}")

                # Reset distance tracking for title display
                min_dist_centroid_right = float('inf')
                min_dist_centroid_left = float('inf')
                
                # Bbox size reduction factor
                BBOX_SIZE_FACTOR = 0.85
                
                # First pass: find the closest object along each pointing ray
                closest_obj_right = None
                closest_obj_left = None
                closest_dist_along_ray_right = float('inf')
                closest_dist_along_ray_left = float('inf')
                
                if pointing_data is not None:
                    gesture_class = pointing_data.get("gesture_class", 0)
                    
                    for point_idx, point_3d in enumerate(point_3d_list):
                        # Get centroid for this object
                        if (bbox_3d_list and point_idx < len(bbox_3d_list) 
                            and bbox_3d_list[point_idx] is not None):
                            bbox_info = bbox_3d_list[point_idx]
                            corners = bbox_info.get("corners", {})
                            if "centroid" in corners:
                                obj_centroid = np.array(corners["centroid"])
                            else:
                                obj_centroid = np.array(point_3d)
                        else:
                            obj_centroid = np.array(point_3d)
                        
                        track_id = track_ids[point_idx] if point_idx < len(track_ids) else point_idx
                        
                        # Get bbheight for this object (use cached or default)
                        if track_id in prism_dimensions_cache:
                            obj_bbheight = prism_dimensions_cache[track_id].get("height", 0.5) * BBOX_SIZE_FACTOR
                        else:
                            obj_bbheight = 0.5 * BBOX_SIZE_FACTOR  # default threshold
                        
                        # Check right arm pointing (gesture_class 2 or 3)
                        if gesture_class in [2, 3]:
                            right_start = pointing_data["right_start"]
                            right_end = pointing_data["right_end"]
                            perp_dist = distancia_ponto_para_reta_3d(obj_centroid, right_start, right_end)
                            
                            if perp_dist < obj_bbheight and perp_dist != float('inf'):
                                # Calculate distance along the ray (how far from the hand)
                                v = right_end - right_start
                                w = obj_centroid - right_start
                                dist_along = np.dot(w, v) / np.linalg.norm(v)
                                
                                if dist_along > 0 and dist_along < closest_dist_along_ray_right:
                                    closest_dist_along_ray_right = dist_along
                                    closest_obj_right = point_idx
                        
                        # Check left arm pointing (gesture_class 1 or 3)
                        if gesture_class in [1, 3]:
                            left_start = pointing_data["left_start"]
                            left_end = pointing_data["left_end"]
                            perp_dist = distancia_ponto_para_reta_3d(obj_centroid, left_start, left_end)
                            
                            if perp_dist < obj_bbheight and perp_dist != float('inf'):
                                # Calculate distance along the ray (how far from the hand)
                                v = left_end - left_start
                                w = obj_centroid - left_start
                                dist_along = np.dot(w, v) / np.linalg.norm(v)
                                
                                if dist_along > 0 and dist_along < closest_dist_along_ray_left:
                                    closest_dist_along_ray_left = dist_along
                                    closest_obj_left = point_idx

                # Log experiment data if enabled
                if experiment_log_enabled and pointing_data is not None:
                    gesture_class = pointing_data.get("gesture_class", 0)
                    
                    # Determine selected object (prefer right arm if both are pointing)
                    selected_idx = None
                    selected_hand = None
                    if closest_obj_right is not None:
                        selected_idx = closest_obj_right
                        selected_hand = "right"
                    elif closest_obj_left is not None:
                        selected_idx = closest_obj_left
                        selected_hand = "left"
                    
                    # Get pointing ray info based on gesture
                    right_start = pointing_data.get("right_start")
                    right_end = pointing_data.get("right_end")
                    left_start = pointing_data.get("left_start")
                    left_end = pointing_data.get("left_end")
                    
                    log_entry = {
                        "frame": frame_number,
                        "timestamp": time.time(),
                        "gesture_class": gesture_class,
                        "selected_obj_idx": selected_idx,
                        "selected_obj_id": track_ids[selected_idx] if selected_idx is not None and selected_idx < len(track_ids) else None,
                        "selected_hand": selected_hand,
                        "selected_position_x": point_3d_list[selected_idx][0] if selected_idx is not None else None,
                        "selected_position_y": point_3d_list[selected_idx][1] if selected_idx is not None else None,
                        "selected_position_z": point_3d_list[selected_idx][2] if selected_idx is not None else None,
                        "perpendicular_dist_right": closest_dist_along_ray_right if closest_obj_right is not None else -1,
                        "perpendicular_dist_left": closest_dist_along_ray_left if closest_obj_left is not None else -1,
                        "dist_along_ray_right": closest_dist_along_ray_right if closest_dist_along_ray_right != float('inf') else -1,
                        "dist_along_ray_left": closest_dist_along_ray_left if closest_dist_along_ray_left != float('inf') else -1,
                        "num_objects_in_scene": len(point_3d_list),
                        "right_start_x": right_start[0] if right_start is not None else None,
                        "right_start_y": right_start[1] if right_start is not None else None,
                        "right_start_z": right_start[2] if right_start is not None else None,
                        "right_end_x": right_end[0] if right_end is not None else None,
                        "right_end_y": right_end[1] if right_end is not None else None,
                        "right_end_z": right_end[2] if right_end is not None else None,
                        "left_start_x": left_start[0] if left_start is not None else None,
                        "left_start_y": left_start[1] if left_start is not None else None,
                        "left_start_z": left_start[2] if left_start is not None else None,
                        "left_end_x": left_end[0] if left_end is not None else None,
                        "left_end_y": left_end[1] if left_end is not None else None,
                        "left_end_z": left_end[2] if left_end is not None else None,
                    }
                    experiment_log.append(log_entry)
                    logging.debug(f"Logged experiment entry for frame {frame_number}")

                # Plot each object with improved styling
                for point_idx, point_3d in enumerate(point_3d_list):
                    track_id = (
                        track_ids[point_idx]
                        if point_idx < len(track_ids)
                        else point_idx
                    )

                    # Get class info if available
                    class_id = None
                    if sorted_class_ids and point_idx < len(sorted_class_ids):
                        class_id = sorted_class_ids[point_idx]

                    # Use consistent color based on track ID
                    if track_id in track_colors:
                        color_rgb = track_colors[track_id]
                        color_rgb_norm = utils.normalize_rgb_color(color_rgb)
                    else:
                        # Fallback color using existing approach
                        color_idx = track_id % len(PLOT_COLORS)
                        color_rgb_norm = PLOT_COLORS[color_idx]

                    # Get marker for this class (same as plot_from_json.py)
                    marker = class_markers.get(
                        int(class_id) if class_id is not None else 0, "o"
                    )

                    # Create legend label with class info (same format as plot_from_json.py)
                    if class_id is not None:
                        class_name = CLASS_NAMES.get(
                            int(class_id), f"Class {class_id}"
                        )
                        legend_label = f"ID {track_id}: {class_name}"
                    else:
                        legend_label = f"ID {track_id}: Unknown"

                    # Add object marker with class-specific marker
                    # scatter = ax_3d.scatter(
                    #     point_3d[0],
                    #     point_3d[1],
                    #     point_3d[2],
                    #     c=[color_rgb_norm],
                    #     s=150,
                    #     marker=marker,  # Use class-specific marker
                    #     edgecolors="black",
                    #     linewidths=1.5,
                    #     alpha=0.8,
                    #     zorder=10,
                    # )

                    # Add to legend elements for custom legend positioning
                    from matplotlib.lines import Line2D

                    legend_elements.append(
                        Line2D(
                            [0],
                            [0],
                            marker=marker,
                            color="w",
                            markerfacecolor=color_rgb_norm,
                            markersize=10,
                            markeredgecolor="black",
                            markeredgewidth=1.5,
                            label=legend_label,
                        )
                    )

                    # Plot trajectory with consistent color
                    if (
                        use_sort
                        and track_id in trajectories
                        and len(trajectories[track_id]) > 1
                    ):
                        trajectory = np.array(trajectories[track_id])
                        ax_3d.plot(
                            trajectory[:, 0],
                            trajectory[:, 1],
                            trajectory[:, 2],
                            c=color_rgb_norm,
                            alpha=0.8,
                            linewidth=2.5,
                            zorder=5,
                        )

                        # Add vertical line with consistent color
                        ax_3d.plot(
                            [point_3d[0], point_3d[0]],
                            [point_3d[1], point_3d[1]],
                            [0, point_3d[2]],
                            "--",
                            color=color_rgb_norm,
                            alpha=0.5,
                            linewidth=1,
                            zorder=4,
                        )

                    # Plot 3D footprint (quadrilateral from triangulated bbox corners)
                    if (
                        bbox_3d_list
                        and point_idx < len(bbox_3d_list)
                        and bbox_3d_list[point_idx] is not None
                    ):
                        bbox_info = bbox_3d_list[point_idx]
                        corners = bbox_info.get("corners", {})

                        # Use triangulated centroid (more stable)
                        if "centroid" in corners:
                            centroid = np.array(corners["centroid"])
                        else:
                            # Fallback: use the main triangulated point_3d
                            centroid = np.array(point_3d)

                        # Use cached dimensions or compute them for the first time
                        try:
                            if track_id in prism_dimensions_cache:
                                # Use previously computed dimensions
                                width_estimate = prism_dimensions_cache[track_id]["width"]
                                height_estimate = prism_dimensions_cache[track_id]["height"]
                                depth_estimate = prism_dimensions_cache[track_id]["depth"]
                            else:
                                # First time: estimate dimensions and cache them
                                # Estimate width (horizontal distance between corners)
                                if "bottom_left" in corners and "bottom_right" in corners:
                                    bl = np.array(corners["bottom_left"])
                                    br = np.array(corners["bottom_right"])
                                    width_estimate = np.linalg.norm(br[:2] - bl[:2])  # X and Y only
                                else:
                                    width_estimate = 0.4  # default
                                
                                # Estimate height (Z difference between top and bottom)
                                if "top_center" in corners and "bottom_center" in corners:
                                    tc = np.array(corners["top_center"])
                                    bc = np.array(corners["bottom_center"])
                                    height_estimate = abs(tc[2] - bc[2])
                                else:
                                    height_estimate = bbox_info.get("height", 1.0)
                                
                                # Depth similar to width
                                depth_estimate = width_estimate
                                
                                # Save to cache for future use
                                prism_dimensions_cache[track_id] = {
                                    "width": width_estimate,
                                    "height": height_estimate,
                                    "depth": depth_estimate
                                }
                                logging.info(f"Dimensões fixadas para track_id={track_id}: width={width_estimate:.2f}, height={height_estimate:.2f}")
                            
                            # Generate a prism centered on the centroid (with size reduction factor)
                            faces, bbheight = generate_prism_from_centroid(
                                centroid,
                                width=width_estimate * BBOX_SIZE_FACTOR,
                                depth=depth_estimate * BBOX_SIZE_FACTOR,
                                height=height_estimate * BBOX_SIZE_FACTOR,
                                min_size=0.3
                            )
                            
                            # Determine if this object is being pointed at
                            # Only highlight the CLOSEST object along the pointing ray
                            bbcolor = "yellow"  # Default color
                            if pointing_data is not None:
                                # Calculate distance from centroid to pointing rays
                                obj_dist_right = distancia_ponto_para_reta_3d(
                                    centroid,
                                    pointing_data["right_start"],
                                    pointing_data["right_end"],
                                )
                                obj_dist_left = distancia_ponto_para_reta_3d(
                                    centroid,
                                    pointing_data["left_start"],
                                    pointing_data["left_end"],
                                )
                                
                                # Track minimum distances for title display
                                if obj_dist_right < min_dist_centroid_right:
                                    min_dist_centroid_right = obj_dist_right
                                if obj_dist_left < min_dist_centroid_left:
                                    min_dist_centroid_left = obj_dist_left
                                
                                # Only highlight if this is the CLOSEST object being pointed at
                                if point_idx == closest_obj_right or point_idx == closest_obj_left:
                                    bbcolor = "red"
                                    logging.debug(f"Object {track_id} is the closest object being pointed at!")
                                    
                                    # Publish the 3D footprint of the pointed object (in separate try block)
                                    if send_to_ros_flag and point_3d_list:
                                        try:
                                            if should_publish(
                                                track_id,
                                                point_3d,
                                                last_sended_coordinates,
                                                threshold=COORD_CHANCHE_THRESHOLD,
                                            ):
                                                
                                                update_publish_cache(
                                                    track_id,
                                                    point_3d,
                                                    last_sended_coordinates,
                                                )
                                                print(f"cache outside: {last_sended_coordinates}")
                                                
                                                send_object_footprint_to_ros(point_3d_list[point_idx], topic="ros.object_footprint")
                                        except Exception as ros_e:
                                            logging.warning(f"Failed to send to ROS: {ros_e}")
                            
                            # Convert color name to RGB normalized (0-1)
                            rgb_base = mcolors.to_rgb(bbcolor)
                            face_color = (*rgb_base, 0.25)
                            edge_color = (*rgb_base, 0.8)
                            
                            poly = Poly3DCollection(
                                faces,
                                facecolors=[face_color],
                                edgecolors=[edge_color],
                                linewidths=1.5,
                                zorder=2,
                            )
                            ax_3d.add_collection3d(poly)
                            
                            scatter = ax_3d.scatter(
                                centroid[0],
                                centroid[1],
                                centroid[2],
                                c=[color_rgb_norm],
                                s=40,
                                marker=marker,
                                edgecolors="black",
                                linewidths=1.5,
                                alpha=0.8,
                                zorder=10,
                            )

                        except Exception as e:
                            logging.warning(
                                f"Could not draw prism for track_id={track_id}: {e}"
                            )

                # Configure 3D plot appearance
                ax_3d.set_xlim([-4, 4])
                ax_3d.set_ylim([-4, 4])
                ax_3d.set_zlim([0, 4])

                # Add title with tracking information (use min distances across all objects)
                dist_right_display = min_dist_centroid_right if min_dist_centroid_right != float('inf') else 0.0
                dist_left_display = min_dist_centroid_left if min_dist_centroid_left != float('inf') else 0.0
                title_text = f"Min Dist Right: {dist_right_display:.3f} | Min Dist Left: {dist_left_display:.3f}"
                # if use_sort:
                #     title_text += (
                #         f" - 3D Tracking ({len(point_3d_list)} objects)"
                #     )
                # else:
                #     title_text += (
                #         f" - Raw Triangulation ({len(point_3d_list)} objects)"
                #     )

                ax_3d.set_title(
                    title_text,
                    fontsize=14,
                    color="black",
                    fontweight="bold",
                    pad=3,
                )

                # Add custom legend with same styling as plot_from_json.py
                if legend_elements:
                    legend = ax_3d.legend(
                        handles=legend_elements,
                        loc="upper left",
                        bbox_to_anchor=(0.01, 0.99),
                        fontsize=11,
                        frameon=True,
                        fancybox=False,
                        shadow=False,
                    )

            # Update graph visualization if requested
            if show_plot and show_graph and graph_ax is not None:
                # Update node positions for persistent layout
                for node in graph.nodes():
                    if node not in pos:
                        cam = int(node.split("cam")[1].split("id")[0])
                        obj_id = int(node.split("id")[1])
                        x = (cam - 1) * 6  # Increased horizontal spacing
                        y = obj_id * 3  # Increased vertical spacing
                        pos[node] = (x, y)

                # Visualize the graph
                visualize_graph(
                    graph, graph_ax, frame_number, pos, node_color_map
                )

            # Update figure and handle events if plotting is enabled
            if show_plot:
                fig.canvas.draw()
                fig.canvas.flush_events()

                # Small delay for smooth visualization
                plt.pause(0.01)
                final_frame = utils.fig_to_image(fig)

                # Save the video frame if requested
                if save_video or (show_plot and frame_number == 0):
                    # Track frame timestamps for FPS estimation
                    frame_timestamps.append(current_time)
                    
                    # Keep only recent timestamps for rolling average
                    if len(frame_timestamps) > fps_window_size:
                        frame_timestamps = frame_timestamps[-fps_window_size:]
                    
                    # Estimate FPS from frame timing (need at least 2 frames)
                    if len(frame_timestamps) >= 2:
                        time_span = frame_timestamps[-1] - frame_timestamps[0]
                        if time_span > 0:
                            estimated_fps = (len(frame_timestamps) - 1) / time_span
                            # Clamp FPS to reasonable range
                            estimated_fps = max(1.0, min(30.0, estimated_fps))
                    
                    # Initialize the video writer after we have a good FPS estimate
                    # Wait for at least 10 frames to get a stable estimate
                    if video_writer is None and len(frame_timestamps) >= 10:
                        logging.info(f"Initializing video writer with estimated FPS: {estimated_fps:.2f}")
                        video_writer = cv2.VideoWriter(
                            output_video_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            estimated_fps,
                            (final_frame.shape[1], final_frame.shape[0]),
                        )

                    if video_writer is not None:
                        final_frame_rgb = cv2.cvtColor(
                            final_frame, cv2.COLOR_BGR2RGB
                        )
                        video_writer.write(final_frame_rgb)

                # Check for exit condition if plotting is enabled
                if not plt.fignum_exists(fig.number):
                    break
            # elif frame_number % 10 == 0:  # Report progress periodically when not plotting
            #     logging.info(f"Processing frame {frame_number}/{video_loader.get_number_of_frames()}")

            t_visualization_end = time.perf_counter()

            # Collect benchmark data for this frame
            if benchmark_enabled:
                t_total = t_visualization_end - t_frame_start
                t_collection = t_collection_end - t_frame_start
                t_detection = t_detection_end - t_detection_start
                t_matching = t_matching_end - t_matching_start
                t_triangulation = t_triangulation_end - t_triangulation_start
                t_tracking = t_tracking_end - t_tracking_start
                t_publish = t_publish_end - t_publish_start
                t_visualization = t_visualization_end - t_visualization_start
                fps_instant = 1.0 / t_total if t_total > 0 else 0.0

                benchmark_entry = {
                    "frame": frame_number,
                    "timestamp": time.time(),
                    "total_latency_ms": t_total * 1000,
                    "frame_collection_ms": t_collection * 1000,
                    "detection_ms": t_detection * 1000,
                    "matching_ms": t_matching * 1000,
                    "triangulation_ms": t_triangulation * 1000,
                    "tracking_ms": t_tracking * 1000,
                    "publishing_ms": t_publish * 1000,
                    "visualization_ms": t_visualization * 1000,
                    "fps_instant": fps_instant,
                    "num_detections": len(detections),
                    "num_triangulated": len(triangulated_points),
                    "num_tracked": len(point_3d_list),
                }
                benchmark_log.append(benchmark_entry)

                if frame_number % 50 == 0:
                    logging.info(
                        f"[Benchmark] Frame {frame_number}: "
                        f"total={t_total*1000:.1f}ms, "
                        f"detect={t_detection*1000:.1f}ms, "
                        f"match={t_matching*1000:.1f}ms, "
                        f"triang={t_triangulation*1000:.1f}ms, "
                        f"track={t_tracking*1000:.1f}ms, "
                        f"viz={t_visualization*1000:.1f}ms, "
                        f"FPS={fps_instant:.1f}"
                    )

            frame_number += 1

        else:
            # Handle incomplete frame collection
            missing_cameras = [
                cam_numbers[i]
                for i, collected in enumerate(frame_collected)
                if not collected
            ]
            logging.warning(
                f"Frame {frame_number}: Only collected {valid_frames}/{len(cam_numbers)} frames. Missing cameras: {missing_cameras}"
            )

            # Check for camera health issues
            for cam in missing_cameras:
                if cam in last_frame_time:
                    time_since_last = current_time - last_frame_time[cam]
                    if time_since_last > 5.0:  # 5 seconds without frames
                        logging.error(
                            f"Camera {cam} appears to be offline (no frames for {time_since_last:.1f}s)"
                        )

            # Small delay before retrying
            time.sleep(0.1)
            continue

    # if frame_number == 478:
    #     quit()
    
    # Save experiment data to CSV if enabled
    if experiment_log_enabled and experiment_log:
        try:
            with open(experiment_output_file, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=experiment_log[0].keys())
                writer.writeheader()
                writer.writerows(experiment_log)
            logging.info(f"Saved {len(experiment_log)} experiment entries to {experiment_output_file}")
        except Exception as e:
            logging.error(f"Failed to save experiment data: {e}")
    
    # Save benchmark data and print summary
    if benchmark_enabled and benchmark_log:
        # Save per-frame data to CSV
        try:
            with open(benchmark_output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=benchmark_log[0].keys())
                writer.writeheader()
                writer.writerows(benchmark_log)
            logging.info(f"Saved {len(benchmark_log)} benchmark entries to {benchmark_output_file}")
        except Exception as e:
            logging.error(f"Failed to save benchmark data: {e}")

        # Print summary statistics
        def _stats(values):
            arr = np.array(values)
            return arr.mean(), arr.std(), arr.min(), arr.max()

        n = len(benchmark_log)
        stage_keys = [
            ("total_latency_ms", "Total pipeline latency"),
            ("frame_collection_ms", "Frame collection"),
            ("detection_ms", "YOLO detection"),
            ("matching_ms", "Epipolar matching"),
            ("triangulation_ms", "RANSAC triangulation"),
            ("tracking_ms", "SORT 3D tracking"),
            ("publishing_ms", "Publishing"),
            ("visualization_ms", "Visualization"),
        ]

        logging.info("=" * 70)
        logging.info("PERFORMANCE BENCHMARK SUMMARY")
        logging.info(f"Total frames processed: {n}")
        total_time = sum(e["total_latency_ms"] for e in benchmark_log) / 1000.0
        logging.info(f"Total run time (pipeline only): {total_time:.2f}s")
        avg_fps = n / total_time if total_time > 0 else 0
        logging.info(f"Average FPS: {avg_fps:.2f}")
        logging.info("-" * 70)
        logging.info(f"{'Stage':<25} {'Mean (ms)':>10} {'Std (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
        logging.info("-" * 70)

        for key, label in stage_keys:
            mean, std, vmin, vmax = _stats([e[key] for e in benchmark_log])
            logging.info(f"{label:<25} {mean:>10.2f} {std:>10.2f} {vmin:>10.2f} {vmax:>10.2f}")

        # Detection counts
        mean_det, std_det, min_det, max_det = _stats([e["num_detections"] for e in benchmark_log])
        mean_tri, std_tri, min_tri, max_tri = _stats([e["num_triangulated"] for e in benchmark_log])
        mean_trk, std_trk, min_trk, max_trk = _stats([e["num_tracked"] for e in benchmark_log])
        logging.info("-" * 70)
        logging.info(f"{'Detections/frame':<25} {mean_det:>10.1f} {std_det:>10.1f} {min_det:>10.0f} {max_det:>10.0f}")
        logging.info(f"{'Triangulated/frame':<25} {mean_tri:>10.1f} {std_tri:>10.1f} {min_tri:>10.0f} {max_tri:>10.0f}")
        logging.info(f"{'Tracked/frame':<25} {mean_trk:>10.1f} {std_trk:>10.1f} {min_trk:>10.0f} {max_trk:>10.0f}")
        logging.info("=" * 70)

    # Cleanup
    if show_plot:
        plt.ioff()
        plt.close()

    if video_writer is not None:
        video_writer.release()

    # video_loader.release()
    logging.info("Processing complete")

    # Export figures if requested
    if export_figures and fig is not None:
        if not os.path.exists(figures_output_dir):
            os.makedirs(figures_output_dir)

        # Get the base name of the video path to use in filenames
        video_basename = os.path.basename(os.path.normpath(video_path))

        # Save the entire figure (mosaic, graph, 3D plot)
        # Ensure plots are up-to-date before saving
        fig.canvas.draw()

        # Save video mosaic part if available
        if show_video and video_ax is not None:
            # Create a temporary figure for the video mosaic
            temp_fig_video, temp_ax_video = plt.subplots(
                figsize=(
                    full_mosaic.shape[1] / 100,
                    full_mosaic.shape[0] / 100,
                ),
                facecolor="white",
            )
            temp_ax_video.imshow(full_mosaic)
            temp_ax_video.axis("off")
            mosaic_filename = os.path.join(
                figures_output_dir,
                f"{video_basename}_video_mosaic_last_frame.png",
            )
            temp_fig_video.savefig(
                mosaic_filename,
                dpi=export_dpi,
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(temp_fig_video)
            logging.info(
                f"Video mosaic saved to {mosaic_filename} at {export_dpi} DPI"
            )

        # Save graph part if available
        if show_graph and graph_ax is not None:
            # Create a temporary figure for the graph
            # Extent of the graph_ax
            bbox = graph_ax.get_tightbbox(
                fig.canvas.get_renderer()
            ).transformed(fig.dpi_scale_trans.inverted())
            temp_fig_graph = plt.figure(
                figsize=(bbox.width, bbox.height), facecolor="white"
            )
            temp_ax_graph = temp_fig_graph.add_subplot(111)

            # Redraw the graph content onto the temporary axis
            # This requires access to the graph drawing function and its parameters
            # For simplicity, we\'ll save the relevant part of the existing figure if possible
            # or notify the user that this specific part needs a more direct save method.
            # The ideal way is to have a function that just draws the graph on a given ax.
            # For now, we save the graph_ax portion from the main figure.
            graph_filename = os.path.join(
                figures_output_dir, f"{video_basename}_graph_last_frame.png"
            )
            fig.savefig(
                graph_filename, dpi=export_dpi, bbox_inches=bbox, pad_inches=0
            )
            plt.close(temp_fig_graph)  # Close the temporary figure
            logging.info(
                f"Graph saved to {graph_filename} at {export_dpi} DPI"
            )

        # Save 3D plot part if available
        if show_3d and ax_3d is not None:
            # Create a temporary figure for the 3D plot
            # Extent of the ax_3d
            bbox = ax_3d.get_tightbbox(fig.canvas.get_renderer()).transformed(
                fig.dpi_scale_trans.inverted()
            )
            # Increase the bbox height slightly to prevent cutoff
            bbox = bbox.from_extents(
                bbox.x0, bbox.y0 - 0.1, bbox.x1, bbox.y1 + 0.05
            )
            print(bbox)
            temp_fig_3d = plt.figure(
                figsize=(bbox.width, bbox.height), facecolor="white"
            )  # Ensure background is white
            # Copy the 3D plot to the new figure - this is tricky with 3D plots.
            # A common way is to re-plot. For simplicity, we save the ax_3d portion.
            plot3d_filename = os.path.join(
                figures_output_dir, f"{video_basename}_3d_plot_last_frame.png"
            )
            fig.savefig(
                plot3d_filename,
                dpi=export_dpi,
                bbox_inches=bbox,
                pad_inches=0.1,
            )
            plt.close(temp_fig_3d)  # Close the temporary figure
            logging.info(
                f"3D plot saved to {plot3d_filename} at {export_dpi} DPI"
            )


if __name__ == "__main__":
    main()
