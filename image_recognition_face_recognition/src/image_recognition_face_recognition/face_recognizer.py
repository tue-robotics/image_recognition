from __future__ import annotations

import pickle
import uuid
from dataclasses import dataclass, field
from os import PathLike
from typing import List, Tuple

import numpy as np
import rospy
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.utils.detect_face import extract_box


ALLOWED_DEVICE_TYPES = ["cpu", "cuda"]


@dataclass
class ROI:
    """
    ROI class that holds a 'region of interest' of an image
    """

    x_offset: int
    y_offset: int
    width: int
    height: int


@dataclass
class L2Distance:
    """
    L2 Distance that holds a l2 distance and an associated label

    Args:
        distance: the l2 distance
        label: the label
    """

    distance: float
    label: str


class RecognizedFace:
    def __init__(
        self,
        image: torch.Tensor,
        roi: ROI,
    ):
        """
        A Recognized face in an image

        :param image: The original cropped image
        :param roi: Region of Interest
        """
        self._image: torch.Tensor = image
        self._roi: ROI = roi
        self.l2_distances: List[L2Distance] = []

    @property
    def image(self) -> torch.Tensor:
        return self._image

    @property
    def image_np(self) -> np.ndarray:
        return self._image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    @property
    def roi(self) -> ROI:
        return self._roi

    def __repr__(self):
        return f"RecognizedFace(image.shape={self.image.shape}, l2_distances={self.l2_distances})"


@dataclass
class TrainedFace:
    """
    This class serves as a custom struct to store information of people we recognized
    """

    label: str
    representations: List[torch.Tensor] = field(default_factory=list)


class FaceRecognizer:
    """
    This class handles the recognition using the Facenet model.
    """

    def __init__(self, device: str, distance_threshold: float = 1.0) -> None:
        """
        Constructor for the list which contains the TrainedFace structure
        """
        self._distance_threshold: float = distance_threshold

        self._trained_faces: List[TrainedFace] = []
        
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

        self._device = torch.device(device_type, device_id)
        rospy.loginfo(f"Running on device: {self._device}")
        
        self._mtcnn = MTCNN(
            keep_all=True,
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self._device,
        )
        self._resnet = InceptionResnetV1(
            pretrained="vggface2", device=self._device).eval()

    def _update_with_categorical_distribution(self, recognition: RecognizedFace) -> RecognizedFace:
        """
        Update the recognition with a categorical distribution of the trained faces

        :param recognition: Input recognition
        :return: Output recognition with an updated categorical distribution
        """
        # Try to get a representation of the detected face
        recognition_embedding = None
        # Unsqueeze the image if it is 3D
        [recognition.image.unsqueeze_(0) if recognition.image.dim() == 3 else recognition.image]
        try:
            recognition_embedding = self._get_embedding(
                recognition.image)
        except Exception as e:
            rospy.logerr(f"Error getting the embedding: {e}")

        if recognition_embedding is not None:
            # Calculate the L2 distance between the embedding and all the stored representations.
            recognition.l2_distances = [
                L2Distance(self._get_min_l2_distance(
                    face.representations, recognition_embedding), face.label)
                for face in self._trained_faces
            ]
            recognition.l2_distances.sort(key=lambda x: x.distance)

        if self._distance_threshold > 0 and (not recognition.l2_distances or recognition.l2_distances[0].distance > self._distance_threshold):
            # Store the new face under a new label
            label = self._generate_label()
            trained_face = TrainedFace(label)
            trained_face.representations.append(recognition_embedding)
            self._trained_faces.append(trained_face)
            recognition.l2_distances.insert(0, L2Distance(0, label))

        return recognition

    def _get_recognized_face(self, img, bbox) -> RecognizedFace:
        
        face = self._mtcnn.extract(img, bbox, save_path=None)
        bbox = extract_box(
            img, bbox, self._mtcnn.image_size, self._mtcnn.margin)
        
        x_offset = bbox[0]
        y_offset = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        return RecognizedFace(face, ROI(x_offset, y_offset, w, h))

    def _get_embedding(self, bgr_image: torch.Tensor) -> torch.Tensor:
        """
        Gets the vector of a face in the image

        :param bgr_image: The input image [batch_size, C, H, W]
        :return: The vector embedding
        """
        if isinstance(bgr_image, np.ndarray):
            bgr_image = torch.stack([torch.from_numpy(bgr_image)], dim=0).permute(0, 3, 2, 1).float()
        elif isinstance(bgr_image, torch.Tensor):
            pass
        else:
            raise ValueError("Input image must be a numpy array or a torch tensor")
        return self._resnet(bgr_image.to(self._device)).detach().cpu().numpy()

    @staticmethod
    def _generate_label() -> str:
        """
        Generates a new UUID label

        :return: The new label
        """
        return str(uuid.uuid4())

    def detect(self, img: np.ndarray) -> List[RecognizedFace]:
        """Detect faces in an image

        :param img: input image in bgr
        :return: The detected faces
        """

        bboxes, probs, _ = self._mtcnn.detect(img)

        if bboxes is None or not len(bboxes):
            return []

        # Extract faces from the bounding boxes
        recognized_faces = [self._get_recognized_face(
            img, bbox) for bbox in bboxes]
        recognized_faces = [self._update_with_categorical_distribution(
            recognized_face) for recognized_face in recognized_faces]

        return recognized_faces

    @ staticmethod
    def _get_min_l2_distance(olds: List[np.ndarray], new: np.ndarray) -> float:
        """
        Calculate the minimal l2 distance of a vector list w.r.t. another vector

        :param olds: List of torch.Tensor
        :param new: torch.Tensor
        :return: Minimal l2 distance
        """
        return min([np.linalg.norm(old - new) for old in olds])

    def _get_trained_face_index(self, label: str) -> int:
        """
        Returns the index of the trained face

        :param label: label of the trained face
        :return: the index of the face in the self._trained faces list, -1 if not found
        """
        for idx, f in enumerate(self._trained_faces):
            if f.label == label:
                return idx
        return -1

    def train(self, img: np.ndarray, name: str) -> TrainedFace:
        """
        Trains the face recognizer with a new face (adds a new face to the trained faces)
        """
        embedding = self._get_embedding(img)
        return self._train_impl(embedding, name)

    def _train_impl(self, face_representation: np.ndarray, name: str) -> None:
        """
        Adds a face to the trained faces, creates a vector representation and adds this

        :param image: Embedded representation of the image
        :param name: The label of the face
        """
        index = self._get_trained_face_index(name)
        if index == -1:
            self._trained_faces.append(TrainedFace(name))

        self._trained_faces[index].representations.append(face_representation)
        rospy.loginfo(f"Trained faces:")
        for trained_face in self._trained_faces:
            rospy.loginfo(
                f"Label: {trained_face.label}, Representations: {len(trained_face.representations)}"
            )

    def clear_trained_faces(self):
        """
        Clears all the trained faces
        """
        self._trained_faces = []

    def save_trained_faces(self, file_name: PathLike):
        pickle.dump(self._trained_faces, file_name)

    def restore_trained_faces(self, file_name: PathLike):
        self._trained_faces = pickle.load(file_name)
