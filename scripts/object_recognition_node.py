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
    def __init__(self, db_path, models_path):
        self._bridge = CvBridge()
        self._recognize_srv = rospy.Service('recognize', Recognize, self._recognize_srv_callback)

        # """1. Create a graph from saved GraphDef file """
        # start = rospy.Time.now()
        # # with open(args.model, 'rb') as f:
        # with open('/home/amigo/ros/indigo/system/src/tensorflow_playground/model/classify_image_graph_def.pb',
        #           'rb') as f:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(f.read())
        #     _ = tf.import_graph_def(graph_def, name='')
        # rospy.loginfo("Step {} took {} seconds".format(1, (rospy.Time.now() - start).to_sec()))

    def _recognize_srv_callback(self, req):
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            raise Exception(error_msg)

        cv2.imshow("image", bgr_image)
        cv2.waitKey(1000)

        """1. Create a graph from saved GraphDef file """
        start = rospy.Time.now()
        # with open(args.model, 'rb') as f:
        with open('/home/amigo/ros/indigo/system/src/tensorflow_playground/model/classify_image_graph_def.pb',
                  'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        rospy.loginfo("Step {} took {} seconds".format(1, (rospy.Time.now() - start).to_sec()))

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
            # with open(args.image, 'rb') as f:
            with open('/home/amigo/ros/indigo/system/src/tensorflow_playground/model/cropped_panda.jpg', 'rb') as f:
                predictions = sess.run(result_tensor, {'DecodeJpeg/contents:0': f.read()})
                predictions = np.squeeze(predictions)
            rospy.loginfo("Step {} took {} seconds".format(4, (rospy.Time.now() - start).to_sec()))

            """5. Open output_labels and construct dict from result"""
            start = rospy.Time.now()
            result = {}
            # with open(args.labels, 'rb') as f:
            with open(
                    '/home/amigo/ros/indigo/system/src/tensorflow_playground/model/imagenet_synset_to_human_label_map.txt',
                    'rb') as f:
                labels = f.read().split("\n")
                result = dict(zip(labels, predictions))
            rospy.loginfo("Step {} took {} seconds".format(5, (rospy.Time.now() - start).to_sec()))

        # For now, we only return the best result
        sorted_result = sorted(result.items(), key=operator.itemgetter(1))
        recognition = Recognition()
        recognition.label = sorted_result[-1][0].split('\t')[1]
        rospy.loginfo("Recognition result: {}".format(recognition))
        return {"recognitions": [recognition]}

if __name__ == '__main__':
    rospy.init_node('object_recognition')

    # ToDo: don't hard
    object_recognition = ObjectRecognition('/home/amigo/ros/indigo/system/src/tensorflow_playground/model/'
                                           'classify_image_graph_def.pb', "")
    rospy.spin()
    print "Closing down..."
