#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge, CvBridgeError

from image_recognition_msgs.srv import Recognize, Annotate
from image_recognition_msgs.msg import Recognition, CategoryProbability, CategoricalDistribution
from sensor_msgs.msg import RegionOfInterest
from std_srvs.srv import Empty

import numpy as np
import cv2
import os
from datetime import datetime

# Openface
import dlib
import openface


def _draw_label(img, label, origin):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1

    text = cv2.getTextSize(label, font, scale, thickness)
    p2 = (origin[0] + text[0][0], origin[1] -text[0][1])
    cv2.rectangle(img, origin, p2, (0, 0, 0), -1)
    cv2.putText(img, label, origin, font, scale, (255, 255, 255), thickness, 8)


def _get_roi(bgr_image, detection, factor_x, factor_y):
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


class FaceRecognition:
    def __init__(self, detection, image, factor_x=0.1, factor_y=0.2):
        self.image = _get_roi(image, detection, factor_x, factor_y)
        self.roi = RegionOfInterest()
        self.roi.x_offset = detection.left()
        self.roi.y_offset = detection.top()
        self.roi.width = detection.width()
        self.roi.height = detection.height()
        self.categorical_distribution = CategoricalDistribution()


class TrainedFace:
    def __init__(self, label):
        self.label = label
        self.representations = []


class OpenfaceROS:
    def __init__(self, align_path, net_path, save_images_folder):
        self._bridge = CvBridge()
        self._annotate_srv = rospy.Service('annotate', Annotate, self._annotate_srv)
        self._recognize_srv = rospy.Service('recognize', Recognize, self._recognize_srv)
        self._clear_srv = rospy.Service('clear', Empty, self._clear_srv)

        # Init align and net
        self._align = openface.AlignDlib(align_path)
        self._net = openface.TorchNeuralNet(net_path, imgDim=96, cuda=False)
        self._face_detector = dlib.get_frontal_face_detector()
        self._trained_faces = []

        if save_images_folder and not os.path.exists(save_images_folder):
            os.makedirs(save_images_folder)

        self._save_images_folder = save_images_folder

    def update_with_categorical_distribution(self, recognition):
        if self._trained_faces:

            # Initialize the categorical distribution with unknown probability value
            default_value = 1.0 / len(self._trained_faces) # TODO, does this make sense?
            recognition.categorical_distribution.unknown_probability = default_value
            recognition.categorical_distribution.probabilities = [CategoryProbability(label=face.label,
                                                                                      probability=default_value)
                                                                  for face in self._trained_faces]

            # Try to get a representation of the detected face
            recognition_representation = None
            try:
                recognition_representation = self._get_representation(recognition.image)
            except Exception as e:
                rospy.logwarn("Could not get representation of face image but detector found one: %s" % str(e))

            # If we have a representation, update with use of the l2 distance w.r.t. the face dict
            if recognition_representation:
                l2_distances = [_get_min_l2_distance(face.representations, recognition_representation)
                                for face in self._trained_faces]

                # Convert these l2 distances to probabilities
                for i in range(0, len(l2_distances)):
                    recognition.categorical_distribution.probabilities[i].label = self._trained_faces[i].label
                    recognition.categorical_distribution.probabilities[i].probability = l2_distances[i] / sum(l2_distances)

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

    def _get_trained_face_index(self, label):
        for i, f in enumerate(self._trained_faces):
            if f.label is label:
                return i
        return -1

    def _annotate_srv(self, req):
        # Convert to opencv image
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            raise Exception("Could not convert to opencv image: %s" % str(e))

        if self._save_images_folder:
            now = datetime.now()
            cv2.imwrite("%s/%s_annotate_%s.jpeg" % (self._save_images_folder, now.strftime("%Y-%m-%d-%H-%M-%S-%f"),
                                                    req.label), bgr_image)

        try:
            face_representation = self._get_representation(bgr_image)
        except Exception as e:
            raise Exception("Could not get representation of face image: %s" % str(e))
        
        index = self._get_trained_face_index(req.label)
        if index == -1:
            self._trained_faces.append(TrainedFace(req.label))

        self._trained_faces[index].representations.append(face_representation)

        rospy.loginfo("Succesfully learned face of '%s'" % req.label)

        return {}

    def _clear_srv(self, req):
        rospy.loginfo("Cleared all faces")
        self._trained_faces = []
        return {}

    def _recognize_srv(self, req):
        # Convert to opencv image
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            raise Exception("Could not convert to opencv image: %s" % str(e))

        # Get face recognitions
        recognitions = [FaceRecognition(d) for d in self._face_detector(bgr_image, 1)]  # 1 = upsample factor

        # Try to add categorical distribution to detections
        recognitions = [self.update_with_categorical_distribution(recognition) for recognition in recognitions]

        # Service response
        return {"recognitions": [Recognition(categorical_distribution=r.categorical_distribution, roi=r.roi)
                                 for r in recognitions]}

if __name__ == '__main__':

    rospy.init_node("face_recognition")

    dlib_shape_predictor_path = rospy.get_param("~align_path")
    openface_neural_network_path = rospy.get_param("~net_path")
    save_images = rospy.get_param("~save_images", False)

    save_images_folder = None
    if save_images:
        save_images_folder = rospy.get_param("~save_images_folder", os.path.expanduser("/tmp/faces"))

    openface_ros = OpenfaceROS(dlib_shape_predictor_path,
                               openface_neural_network_path,
                               save_images_folder)
    rospy.spin()
