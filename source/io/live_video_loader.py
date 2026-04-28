from is_msgs.image_pb2 import Image
from google.protobuf.json_format import Parse
from is_wire.core import Channel, Message, Subscription, ContentType
from google.protobuf.message import Message as PbMessage
from google.protobuf.struct_pb2 import Struct
import socket
from typing import Tuple, Dict, List, Optional
import cv2 
import numpy as np
from source.protobuf.message_pb2 import Detections, Points, Detection3D, BoundingBox3D
from datetime import datetime
from source.messaging.is_to_ros2 import SkeletonPosition

# StreamChannel class for live camera feed
class StreamChannel(Channel):
    def __init__(
        self, uri: str = "amqp://guest:guest@localhost:5672", exchange: str = "is"
    ) -> None:
        super().__init__(uri=uri, exchange=exchange)


    def consume_last(self) -> Tuple[Message, int]:
        """
        Consume the last available message from the channel.
        """
        dropped = 0
        msg = super().consume()
        while True:
            try:
                # will raise an exception when no message remained
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped)

# Function to convert Protocol Buffer Image to NumPy array
def to_np(image: Image) -> np.ndarray:
    """
    Convert a Protocol Buffer Image message to a NumPy array.
    """
    buffer = np.frombuffer(image.data, dtype=np.uint8)
    output = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    return output

# Function to load JSON data
def load_json(filename: str, schema: PbMessage) -> PbMessage:
    """
    Load data from a JSON file and parse it into a Protocol Buffer message.
    """
    with open(file=filename, mode="r", encoding="utf-8") as f:
        proto = Parse(f.read(), schema())
    return proto

def publish(channel: Channel, frame:int, point_3d_list:list, track_ids:list, class_ids:list=None, amqp_url: str = "amqp://10.20.5.2:30000", topic: str = "is.tracker.detections") -> None:
    """
    Publish data to a specified topic.
    
    Args:
        channel: The channel to publish to
        frame: Frame number
        point_3d_list: List of 3D points
        track_ids: List of track IDs
        class_ids: List of class IDs (optional)
        amqp_url: AMQP URL for the message broker
        topic: Topic to publish the message to
    """
    detection = Detections()
    detection.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    detection.frame = frame
   
    # position = SkeletonPosition(amqp_url, topic)

    x = point_3d_list[0][0]
    y = point_3d_list[0][1]

    # print(f"X and Y sent to ROS x{x}, y{y}")
    # position.send_to(f"{str(x)} {str(y)}")
    

    for i, point in enumerate(point_3d_list):
        points = Points()

        points.position.extend(point)
        
        track_id = track_ids[i] if i < len(track_ids) else i
        points.id = int(track_id)
        
        class_id = class_ids[i] if class_ids and i < len(class_ids) else None
        points.name = int(class_id) if class_id is not None else 1
        print(f"Sending: id={points.id}, name={points.name}, class_id_original={class_id}")

        detection.points.append(points)

    message = Message()
    message.content_type = ContentType.PROTOBUF
    message.pack(detection)
    channel.publish(message, topic=topic)


def publish_with_3d_bbox(
    channel: Channel, 
    frame: int, 
    point_3d_list: list, 
    track_ids: list, 
    class_ids: list = None,
    bbox_3d_list: list = None,
    topic: str = "is.tracker.detections"
) -> None:
    """
    Publish 3D detections with bounding boxes (footprint) to a specified topic.
    
    Args:
        channel: The channel to publish to
        frame: Frame number
        point_3d_list: List of 3D center points [[x,y,z], ...]
        track_ids: List of track IDs
        class_ids: List of class IDs (optional)
        bbox_3d_list: List of 3D bounding box info dicts, each containing:
            - 'corners': dict with 'top_left', 'top_right', 'bottom_left', 'bottom_right' (triangulated)
            - 'center': [x, y, z]
            - 'width': float
            - 'height': float
        topic: Topic to publish the message to
    """
    detection = Detections()
    detection.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    detection.frame = frame

    for i, point in enumerate(point_3d_list):
        # Legacy Points message (for backward compatibility)
        points = Points()
        points.position.extend(point)
        track_id = track_ids[i] if i < len(track_ids) else i
        points.id = int(track_id)
        class_id = class_ids[i] if class_ids and i < len(class_ids) else None
        points.name = int(class_id) if class_id is not None else 1
        detection.points.append(points)
        
        # New Detection3D message with 3D bounding box
        det_3d = Detection3D()
        det_3d.id = int(track_id)
        det_3d.class_id = int(class_id) if class_id is not None else 1
        det_3d.position.extend(point)
        
        # Add 3D bounding box footprint if available
        if bbox_3d_list and i < len(bbox_3d_list) and bbox_3d_list[i] is not None:
            bbox_info = bbox_3d_list[i]
            bbox_3d = BoundingBox3D()
            
            # Set center
            if 'center' in bbox_info:
                bbox_3d.center.extend(bbox_info['center'])
            else:
                bbox_3d.center.extend(point)
            
            # Set dimensions
            bbox_3d.width = float(bbox_info.get('width', 0.5))
            bbox_3d.height = float(bbox_info.get('height', 1.7))
            
            # Set footprint from triangulated corners (real 3D data, no fictitious depth)
            corners = bbox_info.get('corners', {})
            if all(k in corners for k in ['bottom_left', 'bottom_right', 'top_left', 'top_right']):
                # Use the actual triangulated corners for the footprint
                # Order: bottom_left, bottom_right, top_right, top_left (clockwise)
                footprint = []
                footprint.extend(corners['bottom_left'])
                footprint.extend(corners['bottom_right'])
                footprint.extend(corners['top_right'])
                footprint.extend(corners['top_left'])
                bbox_3d.footprint.extend(footprint)
                
                # Calculate depth from actual triangulated corners
                bl = np.array(corners['bottom_left'])
                tl = np.array(corners['top_left'])
                depth = np.linalg.norm(tl[:2] - bl[:2])  # Distance in XY plane
                bbox_3d.depth = float(depth)
            
            det_3d.bbox_3d.CopyFrom(bbox_3d)
        
        detection.detections_3d.append(det_3d)
        print(f"Sending 3D: id={det_3d.id}, class={det_3d.class_id}, pos={list(det_3d.position)}, footprint_corners={len(det_3d.bbox_3d.footprint)//3}")

    message = Message()
    message.content_type = ContentType.PROTOBUF
    message.pack(detection)
    channel.publish(message, topic=topic)