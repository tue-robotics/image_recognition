#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError

from image_recognition_msgs.srv import GetFaceProperties
from image_recognition_msgs.msg import FaceProperties
from skybiometry_ros import Skybiometry, SkyFaceProperties

import os
import sys


class SkybiometryFaceProperties:
    def __init__(self, key, secret, timeout, save_images_folder):
        self._bridge = CvBridge()
        self._properties_srv = rospy.Service('get_face_properties', GetFaceProperties, self._get_face_properties_srv)
        self._skybiometry = Skybiometry(key, secret)
        self._api_timeout = timeout

        if save_images_folder:
            self._save_images_folder = os.path.expanduser(save_images_folder)
            if not os.path.exists(self._save_images_folder):
                os.makedirs(self._save_images_folder)
        else:
            self._save_images_folder = None

        rospy.loginfo("SkybiometryFaceProperties initialized:")
        rospy.loginfo(" - api_key=%s", key)
        rospy.loginfo(" - api_secret=%s", secret)
        rospy.loginfo(" - api_timeout=%s", timeout)
        rospy.loginfo(" - save_images_folder=%s", save_images_folder)

    def _get_face_properties_srv(self, req):
        # Convert to opencv images
        try:
            bgr_images = [self._bridge.imgmsg_to_cv2(image, "bgr8") for image in req.face_image_array]
        except CvBridgeError as e:
            raise Exception("Could not convert image to opencv image: %s" % str(e))

        # Call the Skybiometry API
        rospy.loginfo("Trying Skybiometry API request for %d seconds" % self._api_timeout)
        sky_face_properties_array = self._skybiometry.get_face_properties(bgr_images, self._api_timeout)

        face_properties_array = []
        for sky_face_properties in sky_face_properties_array:
            face_properties_array.append(FaceProperties(
                gender=FaceProperties.MALE if sky_face_properties.gender.value == "male" else FaceProperties.FEMALE,
                age=int(sky_face_properties.age_est.value)
            ))

        # Service response
        return {"properties_array": face_properties_array}

if __name__ == '__main__':

    rospy.init_node("face_properties")

    try:
        api_key = rospy.get_param("~api_key", "69efefc20c7f42d8af1f2646ce6742ec")
        api_secret = rospy.get_param("~api_secret", "5fab420ca6cf4ff28e7780efcffadb6c")
        api_timeout = rospy.get_param("~api_timeout", 10)
        save_images = rospy.get_param("~save_images", False)

        save_images_folder = None
        if save_images:
            save_images_folder = rospy.get_param("~save_images_folder", "/tmp/skybiometry")
    except KeyError as e:
        rospy.logerr("Parameter %s not found" % e)
        sys.exit(1)

    openface_ros = SkybiometryFaceProperties(api_key,
                                             api_secret,
                                             api_timeout,
                                             save_images_folder)
    rospy.spin()