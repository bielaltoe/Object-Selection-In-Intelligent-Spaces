"""
This file contains the configuration for the Multi-object triangulation and 3D footprint tracking
for multi-camera systems.
"""
from is_msgs.image_pb2 import HumanKeypoints as HKP

CLASS_NAMES = {
0: 'robot',
1: 'desk',
2: 'ladder',
25: 'umbrella',
56: 'chair',
 }

class Config:

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

    TESTE = {
        0: "NOSE",
        1: "LEFT_EYE",
        2: "RIGHT_EYE",
        3: "LEFT_EAR",
        4: "RIGHT_EAR",
        5: "LEFT_SHOULDER",
        6: "RIGHT_SHOULDER",
        7: "LEFT_ELBOW",
        8: "RIGHT_ELBOW",
        9: "LEFT_WRIST",
        10: "RIGHT_WRIST",
        11: "LEFT_HIP",
        12: "RIGHT_HIP",
        13: "LEFT_KNEE",
        14: "RIGHT_KNEE",
        15: "LEFT_ANKLE",
        16: "RIGHT_ANKLE",
    }

    links = [
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
