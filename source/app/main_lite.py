"""
main_lite.py — Versão simplificada (headless) do pipeline de rastreamento multi-câmera.

Diferenças em relação ao main.py:
  - Sem carregamento de vídeos de arquivo (VideoLoader)
  - Sem visualizações (matplotlib, mosaico de câmeras, gráfico 3D, grafo)
  - Configuração via arquivo JSON (em vez de argparse)
  - Mantém toda a lógica core: IS channels, YOLO, triangulação, SORT 3D, skeleton/gesture, publicação IS

Uso:
    python -m source.app.main_lite config_lite.json

O arquivo JSON deve conter os campos definidos em config_lite.json.
"""

import json
import logging
import os
import socket
import sys
import time

import networkx as nx
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from google.protobuf.struct_pb2 import Struct

from is_msgs.image_pb2 import HumanKeypoints as HKP
from is_msgs.image_pb2 import Image, ObjectAnnotations
from is_wire.core import Message, Subscription

from source.config.config import CLASS_NAMES
from source.core.matcher import Matcher
from source.core.three_dimentional_tracker import SORT_3D
from source.core.tracker import Tracker
from source.core.triangulation import triangulate_ransac
from source.io.io_utils import save_3d_coordinates_with_ids
from source.io.live_video_loader import (
    StreamChannel,
    publish_with_3d_bbox,
    to_np,
)
from source.messaging.publish_to_ros import (
    send_object_footprint_to_ros,
    should_publish,
    update_publish_cache,
)
from source.ml.classifier import Gesture

# ---------------------------------------------------------------------------
# Skeleton keypoint mapping (COCO format) — igual ao main.py
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Global gesture classifier
# ---------------------------------------------------------------------------
gesture_obj = Gesture()

last_sended_coordinates = []
COORD_CHANGE_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Helpers (copiados do main.py, sem dependência de matplotlib)
# ---------------------------------------------------------------------------

def distancia_ponto_para_reta_3d(C, P0, P1):
    """
    Distância de um ponto C a uma semi-reta que começa em P0 e aponta para P1.
    Retorna np.inf se o ponto estiver atrás da origem da semi-reta.
    """
    v = P1 - P0
    w = C - P0
    v_norm_sq = np.dot(v, v)
    if v_norm_sq < 1e-9:
        return np.linalg.norm(w)
    t = np.dot(w, v) / v_norm_sq
    if t <= 0:
        return np.inf
    cross = np.cross(w, v)
    return np.linalg.norm(cross) / np.linalg.norm(v)


def load_config(path: str) -> dict:
    """Carrega e valida o arquivo JSON de configuração."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with open(path, "r") as f:
        cfg = json.load(f)

    # Defaults para campos opcionais
    defaults = {
        "cam_numbers": [0, 1, 2, 3],
        "yolo_model": "models/yolo11x.pt",
        "confidence": 0.6,
        "class_list": [0],
        "distance_threshold": 0.4,
        "drift_threshold": 0.4,
        "reference_point": "bottom_center",
        "max_age": 10,
        "min_hits": 3,
        "dist_threshold": 1.0,
        "use_3d_tracker": True,
        "publish_detections": True,
        "publish_topic": "is.tracker.detections",
        "publish_pointing": True,
        "pointing_topic": "MOTPointing.0.Detection",
        "plot_skeleton": False,
        "save_coordinates": False,
        "output_file": "output.json",
        "send_to_ros": False,
    }
    for key, val in defaults.items():
        cfg.setdefault(key, val)

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # 1. Carrega configuração do JSON
    # ------------------------------------------------------------------
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_lite.json"
    cfg = load_config(config_path)
    logging.info(f"Configuração carregada de: {config_path}")

    address         = cfg["address"]
    cam_numbers     = sorted(set(cfg["cam_numbers"]))
    yolo_model      = cfg["yolo_model"]
    confidence      = cfg["confidence"]
    class_list      = cfg["class_list"]
    distance_threshold = cfg["distance_threshold"]
    drift_threshold = cfg["drift_threshold"]
    reference_point = cfg["reference_point"]
    max_age         = cfg["max_age"]
    min_hits        = cfg["min_hits"]
    dist_threshold  = cfg["dist_threshold"]
    use_sort        = cfg["use_3d_tracker"]
    publish_detections    = cfg["publish_detections"]
    publish_topic        = cfg["publish_topic"]
    publish_pointing     = cfg["publish_pointing"]
    pointing_topic       = cfg["pointing_topic"]
    plot_skeleton   = cfg["plot_skeleton"]
    save_flag       = cfg["save_coordinates"]
    output_json_file = cfg["output_file"]
    send_to_ros_flag = cfg["send_to_ros"]

    amqp_url = f"amqp://guest:guest@{address}"

    logging.info(f"Endereço AMQP: {amqp_url}")
    logging.info(f"Câmeras: {cam_numbers}")
    logging.info(f"Modelo YOLO: {yolo_model}")
    logging.info(f"Ponto de referência: {reference_point}")
    logging.info(f"Publicar detecções no IS: {publish_detections} → {publish_topic}")
    logging.info(f"Skeleton/Gesture: {plot_skeleton}")

    # ------------------------------------------------------------------
    # 2. Inicializa módulos de tracking
    # ------------------------------------------------------------------
    tracker = Tracker(
        [yolo_model for _ in range(len(cam_numbers))],
        cam_numbers,
        class_list,
        confidence,
    )
    matcher = Matcher(distance_threshold, drift_threshold)

    sort_tracker = None
    if use_sort:
        sort_tracker = SORT_3D(
            max_age=max_age, min_hits=min_hits, dist_threshold=dist_threshold
        )
        logging.info("SORT 3D tracker inicializado")

    prism_dimensions_cache = {}

    # ------------------------------------------------------------------
    # 3. Inicializa IS channels — câmeras
    # ------------------------------------------------------------------
    publish_channel = StreamChannel(amqp_url)
    channels = {}
    subscriptions = {}

    for cam_idx in cam_numbers:
        try:
            channels[cam_idx] = StreamChannel(amqp_url)
            subscriptions[cam_idx] = Subscription(
                channels[cam_idx], name=f"CameraCapture{cam_idx}"
            )
            subscriptions[cam_idx].subscribe(topic=f"CameraGateway.{cam_idx}.Frame")
            logging.info(f"Inscrito na câmera {cam_idx}")
        except Exception as e:
            logging.error(f"Falha ao inscrever na câmera {cam_idx}: {e}")

    # ------------------------------------------------------------------
    # 4. IS channel — skeleton (opcional)
    # ------------------------------------------------------------------
    skeleton_channel = None
    if plot_skeleton:
        try:
            skeleton_channel = StreamChannel(amqp_url)
            sk_sub = Subscription(skeleton_channel, name="SkeletonSubscriber")
            sk_sub.subscribe(topic="SkeletonsGrouper.0.Localization")
            logging.info("Inscrito em SkeletonsGrouper.0.Localization")
        except Exception as e:
            logging.error(f"Falha ao inscrever no tópico de skeleton: {e}")
            plot_skeleton = False
            skeleton_channel = None

    # ------------------------------------------------------------------
    # 5. Loop principal (headless)
    # ------------------------------------------------------------------
    frame_number = 0
    last_frame_time = {}
    collection_timeout = 1.0

    logging.info("Iniciando loop principal (headless)...")

    while True:
        frames = []
        current_time = time.time()
        frame_collected = [False] * len(cam_numbers)

        t_frame_start = time.perf_counter()
        start_collection_time = time.time()

        # --- 5.1 Coleta de frames ---
        while (
            not all(frame_collected)
            and (time.time() - start_collection_time) < collection_timeout
        ):
            for cam_idx, cam in enumerate(cam_numbers):
                if frame_collected[cam_idx]:
                    continue
                try:
                    message, dropped = channels[cam].consume_last()
                    if message is not None:
                        image = message.unpack(Image)
                        frame = to_np(image)
                        if frame is not None:
                            while len(frames) <= cam_idx:
                                frames.append(None)
                            frames[cam_idx] = frame
                            frame_collected[cam_idx] = True
                            last_frame_time[cam] = current_time
                            if dropped > 10:
                                logging.warning(
                                    f"Câmera {cam}: {dropped} frames descartados"
                                )
                except socket.timeout:
                    continue
                except Exception as e:
                    logging.error(f"Erro ao ler câmera {cam}: {e}")

        valid_frames = sum(frame_collected)
        if valid_frames < len(cam_numbers):
            missing = [cam_numbers[i] for i, ok in enumerate(frame_collected) if not ok]
            logging.warning(
                f"Frame {frame_number}: apenas {valid_frames}/{len(cam_numbers)} frames. "
                f"Câmeras ausentes: {missing}"
            )
            time.sleep(0.1)
            continue

        logging.info(
            f"Frame {frame_number}: todos os {len(cam_numbers)} frames coletados"
        )

        # --- 5.2 Skeleton / gesture (opcional) ---
        skeleton_pts = None
        if plot_skeleton and skeleton_channel is not None:
            try:
                skeleton_msg, _ = skeleton_channel.consume_last()
                if skeleton_msg is not None:
                    skeleton_results = skeleton_msg.unpack(ObjectAnnotations)
                    n_persons = len(skeleton_results.objects)
                    if n_persons >= 1:
                        pts = np.zeros((n_persons, 17, 3), dtype=np.float64)
                        for i, skeleton in enumerate(skeleton_results.objects):
                            for part in skeleton.keypoints:
                                if part.id in TO_COCO_IDX:
                                    idx = TO_COCO_IDX[part.id]
                                    pts[i, idx, 0] = part.position.x
                                    pts[i, idx, 1] = part.position.y
                                    pts[i, idx, 2] = part.position.z
                        skeleton_pts = pts
                        logging.debug(f"Skeleton recebido: {n_persons} pessoa(s)")
            except socket.timeout:
                pass
            except Exception as e:
                logging.warning(f"Erro ao consumir skeleton: {e}")

        # --- 5.3 Detecção e tracking 2D ---
        graph = nx.Graph()
        tracker.detect_and_track(frames)
        detections = tracker.get_detections()

        triangulated_points = []
        graph_component_ids = []
        node_color_map = {}
        class_ids = []
        bbox_3d_list = []

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

        # --- 5.4 Matching e edges ---
        for k in cam_numbers:
            for j in cam_numbers:
                if k != j and k < j:
                    matches = matcher.match_detections(detections, [k, j])
                    for match in matches:
                        n1 = f"cam{k}id{int(match[0].id)}"
                        n2 = f"cam{j}id{int(match[1].id)}"
                        if n1 in graph.nodes and n2 in graph.nodes:
                            graph.add_edge(n1, n2)

        # --- 5.5 Triangulação ---
        for idx, c in enumerate(nx.connected_components(graph)):
            subgraph = graph.subgraph(c)
            if len(subgraph.nodes) > 1:
                ids = sorted(subgraph.nodes)
                d2_points = []
                proj_matricies = []
                class_id = subgraph.nodes[ids[0]]["name"]

                bbox_corners_2d = {
                    "top_left": [], "top_right": [],
                    "bottom_left": [], "bottom_right": [],
                    "bottom_center": [], "top_center": [], "centroid": [],
                }

                for node in ids:
                    cam = int(node.split("cam")[1].split("id")[0])
                    bbox = subgraph.nodes[node]["bbox"]
                    centroid = subgraph.nodes[node]["centroid"]
                    P_cam = matcher.P_all[cam]

                    if reference_point == "center":
                        point_2d = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)
                    elif reference_point == "top_center":
                        point_2d = ((bbox[2] + bbox[0]) / 2, bbox[1])
                    elif reference_point == "feet":
                        bottom_offset = 0.2 * (bbox[3] - bbox[1])
                        point_2d = ((bbox[2] + bbox[0]) / 2, bbox[3] - bottom_offset)
                    else:  # bottom_center
                        point_2d = ((bbox[2] + bbox[0]) / 2, bbox[3])

                    bbox_corners_2d["top_left"].append(((bbox[0], bbox[1]), P_cam))
                    bbox_corners_2d["top_right"].append(((bbox[2], bbox[1]), P_cam))
                    bbox_corners_2d["bottom_left"].append(((bbox[0], bbox[3]), P_cam))
                    bbox_corners_2d["bottom_right"].append(((bbox[2], bbox[3]), P_cam))
                    bbox_corners_2d["bottom_center"].append(
                        (((bbox[0] + bbox[2]) / 2, bbox[3]), P_cam)
                    )
                    bbox_corners_2d["top_center"].append(
                        (((bbox[0] + bbox[2]) / 2, bbox[1]), P_cam)
                    )
                    bbox_corners_2d["centroid"].append((centroid, P_cam))

                    d2_points.append(point_2d)
                    proj_matricies.append(P_cam)

                if len(d2_points) >= 2:
                    point_3d, _ = triangulate_ransac(proj_matricies, d2_points)

                    bbox_3d_corners = {}
                    for corner_name, corner_data in bbox_corners_2d.items():
                        if len(corner_data) >= 2:
                            corner_pts = [c[0] for c in corner_data]
                            corner_proj = [c[1] for c in corner_data]
                            corner_3d, _ = triangulate_ransac(corner_proj, corner_pts)
                            bbox_3d_corners[corner_name] = corner_3d

                    bbox_3d_info = None
                    if "bottom_center" in bbox_3d_corners and "top_center" in bbox_3d_corners:
                        height_3d = (
                            bbox_3d_corners["top_center"][2]
                            - bbox_3d_corners["bottom_center"][2]
                        )
                        if "bottom_left" in bbox_3d_corners and "bottom_right" in bbox_3d_corners:
                            width_3d = np.linalg.norm(
                                np.array(bbox_3d_corners["bottom_right"][:2])
                                - np.array(bbox_3d_corners["bottom_left"][:2])
                            )
                        else:
                            width_3d = 0.5
                        bbox_3d_info = {
                            "corners": bbox_3d_corners,
                            "center": point_3d,
                            "width": width_3d,
                            "height": abs(height_3d) if height_3d else 1.7,
                        }

                    triangulated_points.append(point_3d)
                    bbox_3d_list.append(bbox_3d_info)
                    graph_component_ids.append(idx)
                    class_ids.append(class_id)

        # --- 5.6 SORT 3D ---
        if use_sort and triangulated_points and sort_tracker is not None:
            sort_result = sort_tracker.update(triangulated_points, class_ids)
            point_3d_list = sort_result["positions"]
            track_ids = sort_result["ids"]
            sorted_class_ids = sort_result["class_ids"]
            trajectories = sort_result["trajectories"]
            logging.info(
                f"Frame {frame_number}: SORT rastreando {len(point_3d_list)} objeto(s)"
            )
        else:
            point_3d_list = triangulated_points
            track_ids = graph_component_ids
            sorted_class_ids = class_ids
            trajectories = {}

        # --- 5.7 Salvar coordenadas (opcional) ---
        if save_flag and point_3d_list:
            save_3d_coordinates_with_ids(
                frame_number, point_3d_list, track_ids, output_json_file, sorted_class_ids
            )

        # --- 5.8 Publicar via IS ---
        if publish_detections and point_3d_list:
            publish_with_3d_bbox(
                channel=publish_channel,
                frame=frame_number,
                point_3d_list=point_3d_list,
                track_ids=track_ids,
                class_ids=sorted_class_ids,
                bbox_3d_list=bbox_3d_list,
                topic=publish_topic,
            )

        # --- 5.9 Lógica de gesture / pointing (sem visualização) ---
        pointing_data = None
        closest_obj_right = None
        closest_obj_left = None
        closest_dist_along_ray_right = float("inf")
        closest_dist_along_ray_left = float("inf")
        BBOX_SIZE_FACTOR = 0.85

        if plot_skeleton and skeleton_pts is not None:
            for person_idx in range(skeleton_pts.shape[0]):
                pts = skeleton_pts[person_idx]
                try:
                    (
                        gesture_class,
                        left_distancia,
                        left_teta,
                        right_distancia,
                        right_teta,
                    ) = gesture_obj.classificador_ml(pts)

                    (
                        rightx, righty, rightz,
                        leftx, lefty, leftz,
                        left_end, right_end,
                    ) = gesture_obj.reta_para_plot(pts, length=5)

                    right_start = np.array([rightx[0], righty[0], rightz[0]])
                    left_start  = np.array([leftx[0],  lefty[0],  leftz[0]])

                    pointing_data = {
                        "gesture_class": gesture_class,
                        "right_start": right_start,
                        "right_end": right_end,
                        "left_start": left_start,
                        "left_end": left_end,
                    }

                    logging.debug(
                        f"Gesture classe={gesture_class} | "
                        f"right_dist={right_distancia:.3f} | left_dist={left_distancia:.3f}"
                    )
                except Exception as e:
                    logging.debug(f"Erro no processamento de gesture: {e}")

        # Determina o objeto mais próximo das semi-retas de apontamento
        if pointing_data is not None:
            gesture_class = pointing_data.get("gesture_class", 0)

            for point_idx, point_3d in enumerate(point_3d_list):
                if (bbox_3d_list and point_idx < len(bbox_3d_list)
                        and bbox_3d_list[point_idx] is not None):
                    corners = bbox_3d_list[point_idx].get("corners", {})
                    obj_centroid = np.array(
                        corners["centroid"] if "centroid" in corners else point_3d
                    )
                else:
                    obj_centroid = np.array(point_3d)

                track_id = track_ids[point_idx] if point_idx < len(track_ids) else point_idx
                obj_bbheight = (
                    prism_dimensions_cache[track_id].get("height", 0.5) * BBOX_SIZE_FACTOR
                    if track_id in prism_dimensions_cache
                    else 0.5 * BBOX_SIZE_FACTOR
                )

                if gesture_class in [2, 3]:
                    perp_dist = distancia_ponto_para_reta_3d(
                        obj_centroid,
                        pointing_data["right_start"],
                        pointing_data["right_end"],
                    )
                    if perp_dist < obj_bbheight and perp_dist != float("inf"):
                        v = pointing_data["right_end"] - pointing_data["right_start"]
                        w = obj_centroid - pointing_data["right_start"]
                        dist_along = np.dot(w, v) / np.linalg.norm(v)
                        if dist_along > 0 and dist_along < closest_dist_along_ray_right:
                            closest_dist_along_ray_right = dist_along
                            closest_obj_right = point_idx

                if gesture_class in [1, 3]:
                    perp_dist = distancia_ponto_para_reta_3d(
                        obj_centroid,
                        pointing_data["left_start"],
                        pointing_data["left_end"],
                    )
                    if perp_dist < obj_bbheight and perp_dist != float("inf"):
                        v = pointing_data["left_end"] - pointing_data["left_start"]
                        w = obj_centroid - pointing_data["left_start"]
                        dist_along = np.dot(w, v) / np.linalg.norm(v)
                        if dist_along > 0 and dist_along < closest_dist_along_ray_left:
                            closest_dist_along_ray_left = dist_along
                            closest_obj_left = point_idx

        # --- Resolve class names for pointed objects ---
        def _class_name(obj_idx):
            """Returns the class name string for an object index, or None."""
            if obj_idx is None:
                return None
            class_id = (
                sorted_class_ids[obj_idx]
                if sorted_class_ids and obj_idx < len(sorted_class_ids)
                else None
            )
            return CLASS_NAMES.get(int(class_id), f"Class {class_id}") if class_id is not None else None

        closest_class_name_right = _class_name(closest_obj_right)
        closest_class_name_left  = _class_name(closest_obj_left)

        # --- Publish MOTPointing.0.Detection ---
        if publish_pointing and plot_skeleton and pointing_data is not None:
            right_text = closest_class_name_right if closest_class_name_right else "None"
            left_text  = closest_class_name_left  if closest_class_name_left  else "None"

            struct_msg = Struct()
            struct_msg.fields["right_pointing"].string_value = right_text
            struct_msg.fields["left_pointing"].string_value  = left_text
            publish_channel.publish(
                Message(content=struct_msg),
                topic=pointing_topic,
            )
            logging.info(
                f"Frame {frame_number}: [{pointing_topic}] "
                f"right={right_text} | left={left_text}"
            )

        # --- Log e ação sobre objeto selecionado ---
        selected_idx = None
        selected_hand = None
        if closest_obj_right is not None:
            selected_idx = closest_obj_right
            selected_hand = "right"
        elif closest_obj_left is not None:
            selected_idx = closest_obj_left
            selected_hand = "left"

        if selected_idx is not None:
            sel_id = track_ids[selected_idx] if selected_idx < len(track_ids) else selected_idx
            sel_pos = point_3d_list[selected_idx]
            logging.info(
                f"Frame {frame_number}: Objeto SELECIONADO id={sel_id} "
                f"mão={selected_hand} pos={[round(v, 3) for v in sel_pos]}"
            )

            if send_to_ros_flag:
                try:
                    if should_publish(sel_id, sel_pos, last_sended_coordinates,
                                      threshold=COORD_CHANGE_THRESHOLD):
                        update_publish_cache(sel_id, sel_pos, last_sended_coordinates)
                        send_object_footprint_to_ros(sel_pos, topic="ros.object_footprint")
                except Exception as ros_e:
                    logging.warning(f"Falha ao enviar para ROS: {ros_e}")
        elif pointing_data is not None and pointing_data.get("gesture_class", 0) != 0:
            logging.info(
                f"Frame {frame_number}: Gesture detectado (classe={pointing_data['gesture_class']}), "
                "nenhum objeto na semi-reta de apontamento"
            )

        frame_number += 1


if __name__ == "__main__":
    main()
