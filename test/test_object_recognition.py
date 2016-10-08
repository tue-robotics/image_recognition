#!/usr/bin/env python

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import rospy
import cv2
import sys

from object_recognition_srvs.srv import Recognize
from std_srvs.srv import Empty

bridge = CvBridge()

rospy.init_node('test_object_recognition')

recognize_srv_name = "recognize"

rospy.loginfo("Waiting for services '%s'" % recognize_srv_name)

rospy.wait_for_service(recognize_srv_name)

recognize_srv = rospy.ServiceProxy(recognize_srv_name, Recognize)


def callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    cv2.imshow("Image window", cv_image)
    key = cv2.waitKey(10)

    if key != -1:
        try:
            print recognize_srv(image=data)
        except:
            pass

    return

# Usage: run a camera node, e.g.:
# - rosrun usb_cam usb_cam_node
# Start this node with the right topic, in this case:
# - rosrun object_recognition_srvs test_object_recognition image:=usb_cam/image_raw

image_sub = rospy.Subscriber("image", Image, callback)
rospy.loginfo("Listening to %s -- spinning .." % image_sub.name)
rospy.loginfo("Press any key to recognize")

rospy.spin()
