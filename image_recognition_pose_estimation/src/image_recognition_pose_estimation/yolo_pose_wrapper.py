import logging
import os
import re
import sys
from typing import List

import numpy as np
from image_recognition_msgs.msg import CategoricalDistribution, CategoryProbability, Recognition
from sensor_msgs.msg import RegionOfInterest
from ultralytics import YOLO
from ultralytics.engine.results import Results


YOLO_POSE_PATTERN = re.compile(r"^yolov8(?:([nsml])|(x))-pose(?(2)-p6|)?.pt$")


class YoloPoseWrapper:
    def __init__(self, model_name: str = "yolov8n-pose.pt", verbose: bool = False):
        if not YOLO_POSE_PATTERN.match(model_name):
            raise ValueError(f"Model name '{model_name}' does not match pattern '{YOLO_POSE_PATTERN.pattern}'")

        self._model = YOLO(model=model_name, task="pose", verbose=verbose)

    def detect_poses(self, image: np.ndarray):
        # Detect poses
        results: List[Results] = self._model.__call__(image)
        recognitions = []

        if keypoints is not None and len(keypoints.shape) == 3:  # If no detections, keypoints will be None
            num_persons, num_bodyparts, _ = keypoints.shape
            for person_id in range(0, num_persons):
                for body_part_id in range(0, num_bodyparts):
                    body_part = self._model["body_parts"][body_part_id]
                    x, y, probability = keypoints[person_id][body_part_id]
                    if probability > 0:
                        recognitions.append(
                            Recognition(
                                group_id=person_id,
                                roi=RegionOfInterest(width=1, height=1, x_offset=int(x), y_offset=int(y)),
                                categorical_distribution=CategoricalDistribution(
                                    probabilities=[CategoryProbability(label=body_part, probability=float(probability))]
                                ),
                            )
                        )

        return recognitions, overlayed_image
