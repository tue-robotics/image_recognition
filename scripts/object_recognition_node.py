#!/usr/bin/env python

# System
import operator

# ROS
import rospy

# OpenCV
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Tensorflow
import tensorflow as tf
import numpy as np

# TU/e Robotics
from object_recognition_srvs.srv import Recognize
from object_recognition_srvs.msg import Recognition


class ObjectRecognition:
    """ Performs object recognition using Tensorflow neural networks """
    def __init__(self, db_path, models_path):
        """ Constructor
        :param db_path: string with path + filename (incl. extension) indicating the database location
        :param models_path: string with path + filename (incl. extension) indicating the location of the text file
        with labels etc.
        """
        self._bridge = CvBridge()
        self._recognize_srv = rospy.Service('recognize', Recognize, self._recognize_srv_callback)
        self._do_recognition = False  # Indicates whether a new request has been received and thus recognition must
        # be performed
        self._filename = "/tmp/tf_obj_rec.jpg"  # Temporary file name
        self._models_path = models_path
        self._recognition = None

        """1. Create a graph from saved GraphDef file """
        start = rospy.Time.now()
        with open(db_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        rospy.loginfo("Step {} took {} seconds".format(1, (rospy.Time.now() - start).to_sec()))

    def _recognize_srv_callback(self, req):
        """ Callback function for the recognize. It saves the image on a temporary location and sets _do_recognition
        to True. Subsequently, it waits until the image has been processed (i.e., until _do_recognition is False again)
        and then returns the result
        :param req: object_recognition_srvs.srv.RecognizeRequest
        :return: object_recognition_srvs.srv.RecognizeResponse
        """
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            raise Exception(error_msg)

        cv2.imshow("image", bgr_image)
        cv2.waitKey(1000)

        # Write the image to file
        # ToDo: directly in memory, saves file operations
        cv2.imwrite(filename=self._filename, img=bgr_image)
        size = bgr_image.shape[:2]  # For now, we assume the entire image is the ROI
        self._recognition = Recognition()
        self._recognition.roi.height = size[0]
        self._recognition.roi.width = size[1]
        self._do_recognition = True

        # Wait until the request has been processed and return the result
        r = rospy.Rate(20.0)
        while not rospy.is_shutdown():
            if not self._do_recognition:
                return {"recognitions": [self._recognition]}

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
            rospy.loginfo("Step {} took {} seconds".format(2, (rospy.Time.now() - start).to_sec()))

            """3. Get result tensor"""
            start = rospy.Time.now()
            result_tensor = sess.graph.get_tensor_by_name("softmax:0")
            rospy.loginfo("Step {} took {} seconds".format(3, (rospy.Time.now() - start).to_sec()))

            """4. Open Image and perform prediction"""
            start = rospy.Time.now()
            predictions = []
            with open(self._filename, 'rb') as f:
                predictions = sess.run(result_tensor, {'DecodeJpeg/contents:0': f.read()})
                predictions = np.squeeze(predictions)
            rospy.loginfo("Step {} took {} seconds".format(4, (rospy.Time.now() - start).to_sec()))

            """5. Open output_labels and construct dict from result"""
            start = rospy.Time.now()
            result = {}
            with open(self._models_path, 'rb') as f:
                labels = f.read().split("\n")
                result = dict(zip(labels, predictions))
            rospy.loginfo("Step {} took {} seconds".format(5, (rospy.Time.now() - start).to_sec()))

        # For now, we only return the best result
        sorted_result = sorted(result.items(), key=operator.itemgetter(1))
        self._recognition.label = sorted_result[-1][0].split('\t')[1]
        rospy.loginfo("Recognition result: {}".format(self._recognition))
        self._do_recognition = False

if __name__ == '__main__':
    rospy.init_node('object_recognition')

    # ToDo: don't hardcode locations
    object_recognition = ObjectRecognition(db_path='/home/amigo/ros/indigo/system/src/tensorflow_playground/model/'
                                           'classify_image_graph_def.pb',
                                           models_path='/home/amigo/ros/indigo/system/src/tensorflow_playground/'
                                                       'model/imagenet_synset_to_human_label_map.txt')
    r = rospy.Rate(20.0)
    while not rospy.is_shutdown():
        object_recognition.update()
        r.sleep()

    print "Closing down..."
