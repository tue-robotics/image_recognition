#!/usr/bin/env python

import numpy as np
import cv2
import os

# Openface
import dlib
import openface


def _get_roi_image(bgr_image, detection, factor_x, factor_y):
    # Get the roi
    min_y = detection.top()
    max_y = detection.bottom()
    min_x = detection.left()
    max_x = detection.right()

    dx = max_x - min_x
    dy = max_y - min_y

    padding_x = int(factor_x * dx)
    padding_y = int(factor_y * dy)

    # Don't go out of bound
    min_y = max(0, min_y - padding_y)
    max_y = min(max_y + padding_y, bgr_image.shape[0]-1)
    min_x = max(0, min_x - padding_x)
    max_x = min(max_x + padding_x, bgr_image.shape[1]-1)

    return bgr_image[min_y:max_y, min_x:max_x]


def _get_min_l2_distance(vector_list_a, vector_b):
    return min([np.dot(vector_a - vector_b, vector_a - vector_b) for vector_a in vector_list_a])


class ROI:
    x_offset = 0
    y_offset = 0
    width = 0
    height = 0

    def __init__(self):
        pass

    def __repr__(self):
        return "(%d, %d, %d, %d)" % (self.x_offset, self.y_offset, self.width, self.height)


class L2Distance:
    def __init__(self, distance, label):
        self.distance = distance
        self.label = label

    def __repr__(self):
        return "L2Distance(%s, %f)" % (self.label, self.distance)


class RecognizedFace:
    def __init__(self, detection, image, factor_x=0.1, factor_y=0.2):
        self.image = _get_roi_image(image, detection, factor_x, factor_y)
        self.roi = ROI()
        self.roi.x_offset = detection.left()
        self.roi.y_offset = detection.top()
        self.roi.width = detection.width()
        self.roi.height = detection.height()
        self.l2_distances = []

    def __repr__(self):
        return "RecognizedFace(roi=%s, l2_distances=%s)" % (self.roi, self.l2_distances)


class TrainedFace:
    def __init__(self, label):
        self.label = label
        self.representations = []


class FaceRecognizer:
    def __init__(self, align_path, net_path):
        # Init align and net
        self._align = openface.AlignDlib(os.path.expanduser(align_path))
        self._net = openface.TorchNeuralNet(os.path.expanduser(net_path), imgDim=96, cuda=False)
        self._face_detector = dlib.get_frontal_face_detector()
        self._trained_faces = []

    def update_with_categorical_distribution(self, recognition):
        if self._trained_faces:
            # Try to get a representation of the detected face
            recognition_representation = None
            try:
                recognition_representation = self._get_representation(recognition.image)
            except Exception as e:
                print "Could not get representation of face image but detector found one: %s" % str(e)

            # If we have a representation, update with use of the l2 distance w.r.t. the face dict
            if recognition_representation is not None:
                recognition.l2_distances = [L2Distance(_get_min_l2_distance(
                    face.representations, recognition_representation), face.label) for face in self._trained_faces]

            # Sort l2 distances probabilities, lowest on index 0
            recognition.l2_distances = sorted(recognition.l2_distances, key=lambda x: x.distance)

        return recognition

    def _get_representation(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        bb = self._align.getLargestFaceBoundingBox(rgb_image)
        if bb is None:
            raise Exception("Unable to find a face in image")

        aligned_face = self._align.align(96, rgb_image, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Unable to align face bb image")

        return self._net.forward(aligned_face)

    def recognize(self, image):

        # Get face recognitions
        recognitions = [RecognizedFace(d, image) for d in self._face_detector(image, 1)]  # 1 = upsample factor

        # Try to add categorical distribution to detections
        recognitions = [self.update_with_categorical_distribution(recognition) for recognition in recognitions]

        return recognitions

    def _get_trained_face_index(self, label):
        for i, f in enumerate(self._trained_faces):
            if f.label is label:
                return i
        return -1

    def train(self, image, name):
        index = self._get_trained_face_index(name)
        if index == -1:
            self._trained_faces.append(TrainedFace(name))

        try:
            face_representation = self._get_representation(image)
        except Exception as e:
            raise Exception("Could not get representation of face image: %s" % str(e))

        self._trained_faces[index].representations.append(face_representation)

    def clear_trained_faces(self):
        self._trained_faces = []
