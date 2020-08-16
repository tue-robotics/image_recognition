import logging
import os
import sys
from image_recognition_msgs.msg import Recognition, CategoricalDistribution, CategoryProbability
from sensor_msgs.msg import RegionOfInterest

from .openpose_models import models


class OpenposeWrapper(object):
    def __init__(self, model_folder, pose_model, net_input_size, net_output_size, num_scales, scale_gap, num_gpu_start,
                 overlay_alpha, python_path=None):
        if python_path:
            sys.path.append(self._validate_dir(python_path))

        try:
            from openpose import pyopenpose as op  # pylint: disable=import-outside-toplevel
            globals()['op'] = op  # Make `op` available globally
        except ImportError as error:
            raise ImportError("{}, please add openpose to your python path using the constructor argument or extend the"
                              " PYTHONPATH environment variable".format(error))

        if pose_model not in models:
            raise ValueError("Pose model not in {}".format(models))

        self._model = models[pose_model]

        model_folder = self._validate_dir(model_folder)
        model_path = os.path.join(model_folder, self._model["path"])
        if not os.path.exists(model_path):
            raise ValueError("Model does not exist on path {}".format(model_path))

        parameters = {
            "logging_level": 3,
            "render_threshold": 0.05,
            "disable_blending": False,
            "model_folder": model_folder,
            "model_pose": str(pose_model),
            "net_resolution": str(net_input_size),
            "output_resolution": str(net_output_size),
            "scale_number": int(num_scales),
            "scale_gap": float(scale_gap),
            "num_gpu_start": int(num_gpu_start),
            "alpha_pose": float(overlay_alpha),
        }

        logging.info("Loading openpose with parameters: %s", parameters)

        self._openpose_wrapper = op.WrapperPython()
        self._openpose_wrapper.configure(parameters)
        self._openpose_wrapper.start()

    @staticmethod
    def _validate_dir(dir_path):
        dir_path = os.path.expanduser(dir_path)
        if not os.path.isdir(dir_path):
            raise ValueError("{} is not a directory!".format(dir_path))
        return dir_path if dir_path[-1] == "/" else dir_path + "/"

    def detect_poses(self, image):
        # `op` added to globals in the constructor
        datum = op.Datum()  # pylint: disable=undefined-variable # noqa: F821
        datum.cvInputData = image
        self._openpose_wrapper.emplaceAndPop([datum])

        keypoints = datum.poseKeypoints
        overlayed_image = datum.cvOutputData

        recognitions = []

        if len(keypoints.shape) == 3:
            num_persons, num_bodyparts, _ = keypoints.shape
            for person_id in range(0, num_persons):
                for body_part_id in range(0, num_bodyparts):
                    body_part = self._model["body_parts"][body_part_id]
                    x, y, probability = keypoints[person_id][body_part_id]
                    if probability > 0:
                        recognitions.append(Recognition(
                            group_id=person_id,
                            roi=RegionOfInterest(
                                width=1,
                                height=1,
                                x_offset=int(x),
                                y_offset=int(y)
                            ),
                            categorical_distribution=CategoricalDistribution(
                                probabilities=[CategoryProbability(label=body_part, probability=float(probability))]
                            )
                        ))

        return recognitions, overlayed_image
