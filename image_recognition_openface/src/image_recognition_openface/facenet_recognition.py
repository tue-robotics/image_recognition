import psutil
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN

# from PIL import Image, ImageDraw
# from IPython import display
# from matplotlib import pyplot as plt
import torch

# import numpy as np
# import pandas as pd
# import os
# import logging

# Set logging level and format
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TrainedFace:
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


class Facenet_recognition:
    def __init__(self):
        """
        Constructor for the list which contains the TrainedFace structure
        """
        self._trained_faces = []

    def print_memory_usage(self, label):
        """
        Prints the memory usage as a metric
        """
        mem_info = psutil.Process().memory_info()
        print(
            f"{label: <10}: vms={mem_info.vms / (1024 * 1024)}, rss={mem_info.rss / (1024 * 1024)}"
        )

    def face_detection(self, img):
        """
        Returns the index of the trained face

        :param img: inpute image
        :return: the bounding boxes of coordinations of the faces it detects
        """
        detector = MTCNN(keep_all=True)
        # IF WE NEED TO LOAD A TEST IMAGE
        # print_memory_usage("init")
        # img = Image.open("/home/iason/ros/noetic/repos/github.com/tue-robotics/image_recognition/image_recognition_openface/test_images/kate_siegel/1.jpg")
        # img = np.asarray(img.copy(), dtype=np.uint8)
        # print_memory_usage(f"#{i}_load")

        boxes, _, landmarks = detector.detect(img, landmarks=True)

        # VISUALIZE THE TEST IMAGE RESULT
        # fig, ax = plt.subplots(figsize=(16, 12))
        # ax.imshow(img)
        # ax.axis('off')

        # for box, landmark in zip(boxes, landmarks):
        # ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
        # ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
        # plt.show()
        return boxes

    def _get_dists(self, embeddings):
        """
        Returns min l2 distance of the identified face (compared with the database of know faces)

        :param embeddings: the embedded representation(s) of a face(s)
        :return: the min distance(s) of the embedded vector compared with the database faces
        :return: the corresponding label(s)
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
                # logging.info(f"{dist_per_emb} dist_per_emb")
                print(dist_per_emb, "dist_per_emb")
                dist_per_emb = []
            dist_per_emb_final.append(dist)
            dist = []

        # logging.info(f"{dist_per_emb_final} dist_per_emb_final")
        print(dist_per_emb_final, "dist_per_emb_final")
        for i in dist_per_emb_final:
            min_of_emb = [min(j) for j in i]
            # logging.info(f"{min_of_emb} min_of_emb")
            print(min_of_emb, "min_of_emb")
            min_of_emb_final.append(min_of_emb)
        # logging.info(f"{min_of_emb_final} min_of_emb_final")
        print(min_of_emb_final, "min_of_emb_final")

        for idx in min_of_emb_final:
            # logging.info(f"{idx} min_index_list_per_emb")
            print(idx, "min_index_list_per_emb")
            min_index_list_per_emb.append(idx.index(min(idx)))
            min_value_list_per_emb.append(min(idx))
            print(min_index_list_per_emb, "min_index_list_per_emb")
            print(min_value_list_per_emb, "min_index_list")

        labelling = [self._trained_faces[i].get_label() for i in min_index_list_per_emb]
        # logging.info(f"{labelling}, {min_index_list}")
        print(labelling, min_value_list_per_emb)

        return min_value_list_per_emb, labelling

    def detection_recognition(self, img, images, labels, save_images, train):
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
        # def detection_recognition(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Running on device: {}".format(device))

        mtcnn = MTCNN(
            keep_all=True,
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
        )

        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        # IF WE NEED TO USE A TEST IMAGE
        # img = Image.open("/home/iason/ros/noetic/repos/github.com/tue-robotics/image_recognition/image_recognition_openface/test_images/angelina_jolie/111.jpg")
        # img = np.asarray(img.copy(), dtype=np.uint8)

        x_aligned = mtcnn(img)
        print(type(x_aligned))
        print(x_aligned.size(), "x_aligned size (1st NN output)")
        x_aligned = x_aligned.cuda()  # add this line
        embeddings = resnet(x_aligned).detach().cpu()
        print(embeddings.size(), type(embeddings), "embeddings size")
        labelss = ["jason", "kona"]

        if not self._trained_faces:
            nam = 0
            for emb in embeddings:
                index = self._get_trained_face_index(labelss[nam])
                if index == -1:
                    self._trained_faces.append(TrainedFace(labelss[nam]))

                self._trained_faces[index].representations.append(emb)
                nam = nam + 1

        # try:
        dist, labelling = self._get_dists(embeddings)
        # if dist > 1:
        print(labelss[0], labelling, "gagagagaga")
        if train:
            idx_label = 0
            for emb in embeddings:
                print(idx_label, "idx_label")
                self.train(emb, labelling[idx_label])
                idx_label = idx_label + 1
        # except:
        # print('we do not have any data yet, lets initialize and please wait on camera')
        print(len(self._trained_faces))
        return dist, labelling

    def _get_trained_face_index(self, label):
        """
        Returns the index of the trained face

        :param label: label of the trained face
        :return: the index of the face in the self._trained faces list
        """
        # TrainedFace(label)
        for i, f in enumerate(self._trained_faces):
            if f.label == label:
                return i
        return -1

    def train(self, roi_image, name):  # IMPLEMENT THIS
        """
        Adds a face to the trained faces, creates a vector representation and adds this

        :param image: Input image
        :param name: The label of the face
        """
        # image = Image.open("/home/iason/ros/noetic/repos/github.com/tue-robotics/image_recognition/image_recognition_openface/test_images/kate_siegel/1.jpg")
        # image = np.asarray(image.copy(), dtype=np.uint8)
        try:
            # face_representation = self.face_detection(roi_image)
            face_representation = roi_image
        except Exception as e:
            raise Exception("Could not get representation of face image: %s" % str(e))

        index = self._get_trained_face_index(name)
        if index == -1:
            print("We do not know this face")
            self._trained_faces.append(TrainedFace(name))

        self._trained_faces[index].representations.append(face_representation)
        print("Trained Faces:")
        for trained_face in self._trained_faces:
            print(
                f"Label: {trained_face.label}, Representations: {len(trained_face.representations)}"
            )
            # print(f"Label: {trained_face.get_label()}, Representations: {trained_face.representations}")


if __name__ == "__main__":
    """
    Runs the whole code. It is not needed if it runs from a ros node.
    """
    face_detection = Facenet_recognition()
    face_detection.detection_recognition()
