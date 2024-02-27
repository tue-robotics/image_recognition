import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from image_recognition_msgs.msg import CategoricalDistribution, CategoryProbability, Recognition
from sensor_msgs.msg import RegionOfInterest
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Yolo pose keypoint labels
# 0: nose
# 1: left-eye
# 2: right-eye
# 3: left-ear
# 4: right-ear
# 5: left-shoulder
# 6: right-shoulder
# 7: left-elbow
# 8: right-elbow
# 9: left-wrist
# 10: right-wrist
# 11: left-hip
# 12: right-hip
# 13: left-knee
# 14: right-knee
# 15: left-ankle
# 16: right-ankle

YOLO_POSE_KEYPOINT_LABELS = [
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-wrist",
    "right-wrist",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
]


YOLO_POSE_PATTERN = re.compile(r"^yolov8(?:([nsml])|(x))-pose(?(2)-p6|)?.pt$")

ALLOWED_DEVICE_TYPES = ["cpu", "cuda"]


class YoloPoseWrapper:
    def __init__(self, model_name: str = "yolov8n-pose.pt", device: str = "cuda:0", verbose: bool = False):
        try:
            device_type, device_id = device.split(":")
        except ValueError:
            if device == "cpu":
                device_type = "cpu"
                device_id = 0
            else:
                raise
        device_id = int(device_id)
        if device_type not in ALLOWED_DEVICE_TYPES:
            raise ValueError(f"Device type '{device_type}' not in {ALLOWED_DEVICE_TYPES}")

        if device_type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available")
            if device_id >= torch.cuda.device_count():
                raise ValueError(
                    f"cuda:{device_id} is not available, only {torch.cuda.device_count()} devices available"
                )

        device = torch.device(device_type, device_id)

        # Validate model name
        model_name_path = Path(model_name)
        model_basename = model_name_path.name
        if not YOLO_POSE_PATTERN.match(model_basename):
            raise ValueError(f"Model name '{model_name}' does not match pattern '{YOLO_POSE_PATTERN.pattern}'")

        if len(model_name_path.parts) == 1:
            model_name = str(Path.home() / "data" / "pytorch_models" / model_name)

        self._model = YOLO(model=model_name, task="pose", verbose=verbose).to(device)

    def detect_poses(self, image: np.ndarray, conf: float = 0.25) -> Tuple[List[Recognition], np.ndarray]:
        # Detect poses
        # This is a wrapper of predict, but we might want to use track
        results: List[Results] = self._model(image, conf=conf)  # Accepts a list

        if not results:
            return [], image

        recognitions = []
        result = results[0]  # Only using
        overlayed_image = result.plot(boxes=False)

        for i, person in enumerate(result.keypoints.cpu().numpy()):
            for j, (x, y, pred_conf) in enumerate(person.data[0]):
                if pred_conf > 0 and x > 0 and y > 0:
                    recognitions.append(
                        Recognition(
                            group_id=i,
                            roi=RegionOfInterest(width=1, height=1, x_offset=int(x), y_offset=int(y)),
                            categorical_distribution=CategoricalDistribution(
                                probabilities=[
                                    CategoryProbability(
                                        label=YOLO_POSE_KEYPOINT_LABELS[j],
                                        probability=float(conf),
                                    )
                                ]
                            ),
                        )
                    )

        return recognitions, overlayed_image
