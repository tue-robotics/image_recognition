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
        self._image = image
        self._roi: ROI = roi
        self.l2_distances: List[L2Distance] = []

    @property
    def image(self) -> np.ndarray:
        # return self._image.permute(1, 2, 0).detach().cpu().numpy()
        return self._image

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

    def __init__(self, distance_threshold: float = 1.0) -> None:
        """
        Constructor for the list which contains the TrainedFace structure
        """
        self._distance_threshold: float = distance_threshold

        self._trained_faces: List[TrainedFace] = []
        self._device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
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
        #return min([np.dot(old - new, old - new) for old in olds])
        return min([np.linalg.norm(old - new) for old in olds])

    def _get_distances(self, embeddings: List[torch.Tensor]) -> Tuple[List[float], List[str]]:
        """
        Returns min l2 distance of the identified face (compared with the database of know faces)

        :param embeddings: The embedded representation(s) of a face(s)
        :return: the min distance(s) of the embedded vector compared with the database faces from the corresponding label(s)
        """
        distance_per_emb_final: List[List] = []
        distance_per_emb = []

        min_of_emb_final = []

        min_index_list_per_emb = []
        min_value_list_per_emb = []

        # Calculate the L2 distance between the embedding and all the stored representations.
        for idx, emb in enumerate(embeddings):  # New measurements
            embeddings.l2_distances = [
                L2Distance(self._get_min_l2_distance(
                    face.representations, emb), face.label)
                for face in self._trained_faces
            ]

            rospy.loginfo(f"{distance_per_emb=} for embedded with index {idx}")
            distance_per_emb_final = [[distance_per_emb]]

        rospy.loginfo(f"{distance_per_emb_final=}")

        # Calculate the minimum distance for each labeled embedding
        # e.g. min distance of all observations of label "Jake"
        for distance in distance_per_emb_final:
            min_of_emb = [min(j) for j in distance]
            rospy.loginfo(f"{min_of_emb} min_of_emb")
            min_of_emb_final.append(min_of_emb)
        rospy.loginfo(f"{min_of_emb_final} min_of_emb_final")

        # Iterate through the minimum distances of every label and find the corresponding index
        for value in min_of_emb_final:
            rospy.loginfo(f"{value} idx")
            min_index_list_per_emb.append(value.index(min(value)))
            min_value_list_per_emb.append(min(value))
            rospy.loginfo(f"{min_index_list_per_emb=}")

        labelling = [
            self._trained_faces[i].label for i in min_index_list_per_emb]
        rospy.loginfo(f"{labelling}, {min_value_list_per_emb}")

        return min_value_list_per_emb, labelling

    def threshold_check(self, dist: List[float], labelling: List[str], labels: List[str], threshold: float) -> None:
        """
        Updates the database with a new addition if the minimum distance is greater than the threshold.

        :param dist: the list of minimum l2 norms of the embedded vectors compared with the database faces
        :param labelling: the corresponding labels of the chosen minimum face representations.
        :param labels: the general list of labels
        :param threshold: the threshold value which denotes if a face is new or not
        """
        for idx, dis in enumerate(dist):
            rospy.loginfo(f"distances are {dist} and labels are {labelling}")
            if dis > threshold:
                # you can always consider the last label or something similar
                labelling[idx] = labels[idx]
                rospy.loginfo(
                    f"Distance is >{threshold} so assign new label: {self._trained_faces[-1].get_label()}, \
                                Representations: {len(self._trained_faces[-1].get_representations())}"
                )
            else:
                rospy.loginfo(f"Distance is <1 so no new label is needed")

    def recognize(self, img: np.ndarray) -> List[RecognizedFace]:
        """
        Returns min l2 distance of the identified face (compared with the database of known faces)

        :param img:
        :return: the min distance(s) of the embedded vector compared with the database faces
        :return: the corresponding label(s)
        """

        bboxes, probs, _ = self._mtcnn.detect(img)

        faces = [self._get_recognized_face(img, bbox) for bbox in bboxes]
        faces_only = [face._image for face in faces]
        faces_only = torch.stack(faces_only, dim=0)
        embeddings = self._get_embedding(faces_only)
        recognized_faces = [RecognizedFace(img, bbox) for bbox in bboxes]
        # rospy.loginfo(
        #     f"{embeddings.size()}, {type(embeddings)}, embeddings size")

        # if not self._trained_faces:
        #     for idx, emb in enumerate(embeddings):
        #         label = labels[idx]
        #         index = self._get_trained_face_index(label)
        #         if index == -1:
        #             self._trained_faces.append(TrainedFace(label, [emb]))
        #         else:
        #             self._trained_faces[index].representations.append(emb)

        # Calculate the L2 norm and check if the distance is bigger than 1 (face that we have not seen yet)
        dist = self._get_distances(embeddings)
        self.threshold_check(dist, labelling, labels, threshold=1)

        rospy.loginfo(f"{len(self._trained_faces)}")
        return dist, labelling

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
      
        #face = self._mtcnn(img)
        #embedding = self._get_embedding(
            #torch.stack([face[0]], dim=0)).squeeze()
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
