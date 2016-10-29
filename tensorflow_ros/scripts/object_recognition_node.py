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
from image_recognition_msgs.msg import Recognition


class ObjectRecognition:
    """ Performs object recognition using Tensorflow neural networks """
    def __init__(self, db_path, models_path, show_images):
        """ Constructor
        :param db_path: string with path + filename (incl. extension) indicating the database location
        :param models_path: string with path + filename (incl. extension) indicating the location of the text file
        with labels etc.
        :param show_images: bool indicating whether to show images as a means of debugging
        """
        # Check if the parameters are correct
        if not (os.path.isfile(db_path) and os.path.isfile(models_path)):
            err_msg = "DB file {} or models file {} does not exist".format(db_path, models_path)
            rospy.logerr(err_msg)
            sys.exit(err_msg)

        self._bridge = CvBridge()
        self._recognize_srv = rospy.Service('recognize', Recognize, self._recognize_srv_callback)
        self._do_recognition = False  # Indicates whether a new request has been received and thus recognition must
        # be performed
        self._filename = "/tmp/tf_obj_rec.jpg"  # Temporary file name
        self._models_path = models_path
        self._recognitions = []  # List with Recognition s
        self._size = {'width': 0, 'height': 0}
        self._show_images = show_images

        """1. Create a graph from saved GraphDef file """
        start = rospy.Time.now()
        with open(db_path, 'rb') as f:
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
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            raise Exception(error_msg)

        if self._show_images:
            cv2.imshow("image", bgr_image)
            cv2.waitKey(1000)

        # Write the image to file
        # ToDo: directly in memory, saves file operations
        cv2.imwrite(filename=self._filename, img=bgr_image)
        size = bgr_image.shape[:2]  # For now, we assume the entire image is the ROI
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
            with open(self._filename, 'rb') as f:
                predictions = sess.run(result_tensor, {'DecodeJpeg/contents:0': f.read()})
                predictions = np.squeeze(predictions)
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
        for res in reversed(sorted_result):
            recognition = Recognition()
            recognition.roi.height = self._size['height']
            recognition.roi.width = self._size['width']
            recognition.label = res[0]
            recognition.probability = res[1]
            self._recognitions.append(recognition)

        rospy.loginfo("\nBest recognition result: {}\nProbability: {}".format(sorted_result[-1][0],
                                                                              sorted_result[-1][1]))
        self._do_recognition = False

if __name__ == '__main__':

    # Start ROS node
    rospy.init_node('object_recognition')

    # Get parameters
    _db_path = rospy.get_param("~database_path")
    _models_path = rospy.get_param("~models_path")
    _show_images = rospy.get_param("~show_image", False)
    rospy.loginfo("\nDB: {}\nModels: {}\nShow image: {}".format(_db_path, _models_path, _show_images))

    # Create object
    object_recognition = ObjectRecognition(db_path=os.path.expanduser(_db_path),
                                           models_path=os.path.expanduser(_models_path),
                                           show_images=_show_images)

    # Start update loop
    r = rospy.Rate(100.0)
    while not rospy.is_shutdown():
        object_recognition.update()
        r.sleep()
