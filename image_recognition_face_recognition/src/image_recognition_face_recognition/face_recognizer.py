from typing import List, Tuple, Union
import rospy
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle

class TrainedFace:
    """
    This class serves as a custom struct to store information of people we recognized
    """
    def __init__(self, label):
        """
        A custom struct to store the names and the embedded representations (tensors) of people
        """
        self.label = label
        self.representations = []

    def get_label(self) -> str:
        """
        A getter for the labels of the struct
        """
        return self.label

    def get_representations(self):
        """
        A getter for the embeddings of the struct
        """
        return self.representations


class FaceRecognizer:
    """
    This class handles the recognition using the Facenet model.
    """
    def __init__(self):
        """
        Constructor for the list which contains the TrainedFace structure
        """
        self._trained_faces = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Running on device: {self.device}")
        self.mtcnn = MTCNN(
            keep_all=True,
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
        )
        
    def face_detection(self, img: np.ndarray) -> List[Union[Tuple[float, float, float, float], None]]:
        """
        Returns the index of the trained face

        :param img: inpute image
        :return: the bounding boxes of coordinations of the faces it detects
        """
   
        # Keep the landmarks for future use
        boxes, _, landmarks = self.mtcnn.detect(img, landmarks=True)
        return boxes

    def _get_dists(self, embeddings: List[torch.Tensor]) -> Tuple[List[float], List[str]]:
        """
        Returns min l2 distance of the identified face (compared with the database of know faces)

        :param embeddings: the embedded representation(s) of a face(s)
        :return: the min distance(s) of the embedded vector compared with the database faces from the corresponding label(s)
        """
        dist_per_emb_final = []
        dist = []
        dist_per_emb = []

        min_of_emb_final = []

        min_index_list_per_emb = []
        min_value_list_per_emb = []

        for e2 in embeddings:
            for e1 in self._trained_faces:
                for e3 in e1.representations:
                    dist_per_emb.append(abs(e3 - e2).norm().item())
                dist.append(dist_per_emb)
                rospy.loginfo(f"{dist_per_emb} dist_per_emb")
                dist_per_emb = []
            dist_per_emb_final.append(dist)
            dist = []

        rospy.loginfo(f"{dist_per_emb_final} dist_per_emb_final")
        for i in dist_per_emb_final:
            min_of_emb = [min(j) for j in i]
            rospy.loginfo(f"{min_of_emb} min_of_emb")
            min_of_emb_final.append(min_of_emb)
        rospy.loginfo(f"{min_of_emb_final} min_of_emb_final")

        for idx in min_of_emb_final:
            rospy.loginfo(f"{idx} idx")
            min_index_list_per_emb.append(idx.index(min(idx)))
            min_value_list_per_emb.append(min(idx))
            rospy.loginfo(f"{min_index_list_per_emb}, min_index_list_per_emb")
            rospy.loginfo(f"{min_value_list_per_emb}, min_index_list")

        labelling = [self._trained_faces[i].get_label() for i in min_index_list_per_emb]
        rospy.loginfo(f"{labelling}, {min_value_list_per_emb}")

        return min_value_list_per_emb, labelling

    def detection_recognition(self, img: np.ndarray, labels: List[str], train: bool) -> Tuple[List[float], List[str]]:
        """
        Returns min l2 distance of the identified face (compared with the database of know faces)

        :param img: the target image collected from camera
        :param images: NA/ a list with all img (why this is needed?)
        :param labels: NA/ a list with all labels (this will be used during integration)
        :param save_images: NA/ a folder with all saved images (why this is needed?)
        :param train: flag to train during inference time
        :return: the min distance(s) of the embedded vector compared with the database faces
        :return: the corresponding label(s)
        """
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        x_aligned = self.mtcnn(img)
        x_aligned = x_aligned.cuda()  # add this line
        embeddings = resnet(x_aligned).detach().cpu()
        rospy.loginfo(f"{embeddings.size()}, {type(embeddings)}, embeddings size")

        if not self._trained_faces:
            for nam, emb in enumerate(embeddings):
                index = self._get_trained_face_index(labels[nam])
                if index == -1:
                    self._trained_faces.append(TrainedFace(labels[nam]))

                self._trained_faces[index].representations.append(emb)

        # try:
        dist, labelling = self._get_dists(embeddings)
        # if dist > 1:
        # in this case we should ask for a label
        rospy.loginfo(f"{labels[0]}, {labelling}, label[0],labelling")

        if train:
            for idx_label, emb in enumerate(embeddings):
                rospy.loginfo(f"{idx_label}, idx_label")
                self.train(emb, labelling[idx_label])
        # except:
        rospy.loginfo(f"{len(self._trained_faces)}")
        return dist, labelling

    def _get_trained_face_index(self, label: str) -> int:
        """
        Returns the index of the trained face

        :param label: label of the trained face
        :return: the index of the face in the self._trained faces list
        """
        for i, f in enumerate(self._trained_faces):
            if f.label == label:
                return i
        return -1

    def train(self, face_representation: np.ndarray, name: str) -> None:
        """
        Adds a face to the trained faces, creates a vector representation and adds this

        :param image: Embedded representation of the image
        :param name: The label of the face
        """
        index = self._get_trained_face_index(name)
        if index == -1:
            rospy.loginfo(f"We do not know this face")
            self._trained_faces.append(TrainedFace(name))

        self._trained_faces[index].representations.append(face_representation)
        rospy.loginfo(f"Trained faces:")
        for trained_face in self._trained_faces:
            rospy.loginfo(
                f"Label: {trained_face.get_label()}, Representations: {len(trained_face.get_representations())}"
            )

    def save_trained_faces(self, file_name):
        pickle.dump(self._trained_faces, file_name)

    def restore_trained_faces(self, file_name):
        self._trained_faces = pickle.load(file_name)
