from is_msgs.image_pb2 import Image
from google.protobuf.json_format import Parse
from is_wire.core import Channel, Message, Subscription, ContentType
from google.protobuf.message import Message as PbMessage
from google.protobuf.struct_pb2 import Struct
import numpy as np
from protobuf.message_pb2 import Detections, Points, Detection3D, BoundingBox3D
from datetime import datetime
from is_to_ros2 import SkeletonPosition
import time

position = SkeletonPosition("amqp://10.20.5.2:30000", "ros.object_footprint")


def send_object_footprint_to_ros(footprint, amqp_url: str = "amqp://10.20.5.2:30000", topic: str = "ros.object_footprint") -> None:
    """
    Publish data to a specified topic.
    
    Args:
        channel: The channel to publish to
        footprint: The 3D footprint coordinates
        amqp_url: AMQP URL for the message broker
        topic: Topic to publish the message to
    """
   

    x = footprint[0]
    y = footprint[1]

    print(f"X e Y indo pro ROS x: {x} , y: {y}")
    
    position.send_to(f"{str(x)} {str(y)}")


def should_publish(track_id, new_coords, cache, threshold=0.1, timeout=10.0):
    """
    Determine if coordinates should be published.
    
    Returns True if:
    - Track ID is new (not in cache)
    - Coordinates changed significantly (distance > threshold)
    - Data is stale (optional timeout exceeded)
    """
    current_time = time.time()
    
    if track_id not in cache:
        return True
    
    old_x, old_y, old_z, last_time = cache[track_id]
    new_x, new_y, new_z = new_coords
    
    # Calculate Euclidean distance
    distance = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
    
    # Publish if moved significantly
    if distance > threshold:
        return True
    
    
    
    return False

def update_publish_cache(track_id, coords, cache):
    """Update the cache with new coordinates."""
    cache[track_id] = (coords[0], coords[1], coords[2], time.time())


if __name__ == "__main__":
    send_object_footprint_to_ros((-1, 0, 0.0),topic="ros.object_footprint")  # Example usage
    time.sleep(1)