#!/usr/bin/env python

import rospy
from object_recognition_srvs.srv import Recognize
from object_recognition_srvs.msg import Recognition
from cv_bridge import CvBridge, CvBridgeError
import cv2

class ObjectRecognition:
    def __init__(self, db_path, models_path):
        self._bridge = CvBridge()
        self._recognize_srv = rospy.Service('recognize', Recognize, self._recognize_srv_callback)

    def _recognize_srv_callback(self, req):
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            error_msg = "Could not convert to opencv image: %s" % e
            rospy.logerr(error_msg)
            raise Exception(error_msg)

        cv2.imshow("image", bgr_image)
        cv2.waitKey(1000)

        return {"recognitions": []}

if __name__ == '__main__':
    rospy.init_node('object_recognition')

    object_recognition = ObjectRecognition("", "")
    rospy.spin()
