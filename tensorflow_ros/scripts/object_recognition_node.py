#!/usr/bin/env python

# System
import os
import operator
import sys

# ROS
import rospy

# OpenCV
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Tensorflow
import tensorflow as tf
import numpy as np

# TU/e Robotics
from image_recognition_msgs.srv import Recognize
from image_recognition_msgs.msg import Recognition, CategoryProbability
from image_recognition_util import image_writer


class TensorflowObjectRecognition:
    """ Performs object recognition using Tensorflow neural networks """
    def __init__(self, graph_path, labels_path, save_images_folder):
        """ Constructor
        :param graph_path: string with path + filename (incl. extension) indicating the database location
        :param labels_path: string with path + filename (incl. extension) indicating the location of the text file
        with labels etc.
        :param save_images_folder: Where to store images for debugging or data collection
        """
        # Check if the parameters are correct
        if not (os.path.isfile(graph_path) and os.path.isfile(labels_path)):
            err_msg = "DB file {} or models file {} does not exist".format(graph_path, labels_path)
            rospy.logerr(err_msg)
            sys.exit(err_msg)

        self._bridge = CvBridge()
        self._recognize_srv = rospy.Service('recognize', Recognize, self._recognize_srv_callback)
        self._do_recognition = False  # Indicates whether a new request has been received and thus recognition must
        # be performed
        self._filename = "/tmp/tf_obj_rec.jpg"  # Temporary file name
        self._models_path = labels_path
        self._recognitions = []  # List with Recognition s
        self._size = {'width': 0, 'height': 0}
        self._save_images_folder = save_images_folder
        self._bgr_image = None

        rospy.loginfo("TensorflowObjectRecognition initialized:")
        rospy.loginfo(" - graph_path=%s", graph_path)
        rospy.loginfo(" - labels_path=%s", labels_path)
        rospy.loginfo(" - save_images_folder=%s", save_images_folder)

        """1. Create a graph from saved GraphDef file """
        start = rospy.Time.now()
        with open(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        rospy.logdebug("Step {} took {} seconds".format(1, (rospy.Time.now() - start).to_sec()))

    def _recognize_srv_callback(self, req):
        """ Callback function for the recognize. It saves the image on a temporary location and sets _do_recognition
        to True. Subsequently, it waits until the image has been processed (i.e., until _do_recognition is False again)
        and then returns the result
        :param req: image_recognition_msgs.srv.RecognizeRequest
        :return: image_recognition_msgs.srv.RecognizeResponse
        """
        try:
            self._bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            raise Exception(error_msg)

        # Write raw image
        if self._save_images_folder:
            image_writer.write_raw(self._save_images_folder, self._bgr_image)

        # Write the image to file
        # ToDo: directly in memory, saves file operations
        cv2.imwrite(filename=self._filename, img=self._bgr_image)
        size = self._bgr_image.shape[:2]  # For now, we assume the entire image is the ROI
        self._size['height'] = size[0]
        self._size['width'] = size[1]
        self._recognitions = []
        self._do_recognition = True

        # Wait until the request has been processed and return the result
        r = rospy.Rate(1000.0)  # Not a problem to spin quickly
        while not rospy.is_shutdown():
            if not self._do_recognition:
                return {"recognitions": self._recognitions}

        # Return an empty result if rospy has been shutdown
        return {"recognitions": []}

    def update(self):
        """ Do the actual work: if _do_recognition is True, it retrieves the saved image and tries to classify it.
        The result is stored in the _recognition member and afterwards _do_recognition is set to False. This function
        is called at a fixed frequency in the mean thread, hence NOT from the service callback. """
        if not self._do_recognition:
            return

        """2. Open tf session"""
        start = rospy.Time.now()
        with tf.Session() as sess:
            rospy.logdebug("Step {} took {} seconds".format(2, (rospy.Time.now() - start).to_sec()))

            """3. Get result tensor"""
            start = rospy.Time.now()
            # result_tensor = sess.graph.get_tensor_by_name("softmax:0")
            result_tensor = sess.graph.get_tensor_by_name("final_result:0")
            rospy.logdebug("Step {} took {} seconds".format(3, (rospy.Time.now() - start).to_sec()))

            """4. Open Image and perform prediction"""
            start = rospy.Time.now()
            predictions = []
            try:
                with open(self._filename, 'rb') as f:
                    predictions = sess.run(result_tensor, {'DecodeJpeg/contents:0': f.read()})
                    predictions = np.squeeze(predictions)
            except Exception as e:
                rospy.logerr("Failed to run tensorflow session: %s", e)

            rospy.logdebug("Step {} took {} seconds".format(4, (rospy.Time.now() - start).to_sec()))

            """5. Open output_labels and construct dict from result"""
            start = rospy.Time.now()
            result = {}
            with open(self._models_path, 'rb') as f:
                labels = f.read().split("\n")
                result = dict(zip(labels, predictions))
            rospy.logdebug("Step {} took {} seconds".format(5, (rospy.Time.now() - start).to_sec()))

        # Sort the results
        sorted_result = sorted(result.items(), key=operator.itemgetter(1))

        # self._recognition.label = sorted_result[-1][0].split('\t')[1]
        recognition = Recognition()
        recognition.roi.height = self._size['height']
        recognition.roi.width = self._size['width']
        recognition.categorical_distribution.unknown_probability = 0.1  # TODO: How do we know this?
        for res in reversed(sorted_result):
            category_probabilty = CategoryProbability(label=res[0], probability=res[1])
            recognition.categorical_distribution.probabilities.append(category_probabilty)

        self._recognitions.append(recognition)

        if sorted_result:
            best_label = sorted_result[-1][0]
            best_prob = sorted_result[-1][1]

            rospy.loginfo("Best recognition result: {} with probability: {}".format(best_label, best_prob))

            # Write unverified annotated image
            if self._save_images_folder:
                image_writer.write_annotated(self._save_images_folder, self._bgr_image, best_label, False)

        self._do_recognition = False

if __name__ == '__main__':

    # Start ROS node
    rospy.init_node('tensorflow_ros')

    try:
        _graph_path = os.path.expanduser(rospy.get_param("~graph_path"))
        _labels_path = os.path.expanduser(rospy.get_param("~labels_path"))
        save_images = rospy.get_param("~save_images", True)

        save_images_folder = None
        if save_images:
            save_images_folder = os.path.expanduser(rospy.get_param("~save_images_folder", "/tmp/tensorflow_ros"))
    except KeyError as e:
        rospy.logerr("Parameter %s not found" % e)
        sys.exit(1)

    # Create object
    object_recognition = TensorflowObjectRecognition(graph_path=_graph_path,
                                                     labels_path=_labels_path,
                                                     save_images_folder=save_images_folder)

    # Start update loop
    r = rospy.Rate(100.0)
    while not rospy.is_shutdown():
        object_recognition.update()
        r.sleep()
