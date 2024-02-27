#! /usr/bin/env python

import os
import socket
import sys
from queue import Empty, Queue

import diagnostic_updater
import rospy
from cv_bridge import CvBridge, CvBridgeError
from image_recognition_msgs.msg import Recognitions
from image_recognition_msgs.srv import Recognize
from image_recognition_util import image_writer
from sensor_msgs.msg import Image

from image_recognition_pose_estimation.yolo_pose_wrapper import YoloPoseWrapper


class PoseEstimationNode:
    def __init__(
        self,
        model_name: str,
        device: str,
        conf: float,
        topic_save_images: bool,
        service_save_images: bool,
        topic_publish_result_image: bool,
        service_publish_result_image: bool,
        save_images_folder: str,
    ):
        """
        Openpose node that wraps the openpose functionality and exposes service and subscriber interfaces

        :param model_name: Model to use
        :param device: Device to use
        :param topic_save_images: Whether we would like to store the (result) images that we receive over topics
        :param service_save_images: Whether we would like to store the (result) images that we receive over topics
        :param topic_publish_result_image: Whether we would like to publish the result images of a topic request
        :param service_publish_result_image: Whether we would like to publish the result images of a service request
        :param save_images_folder: Where to store the images
        """
        self._wrapper = YoloPoseWrapper(model_name, device)

        # We need this q construction because openpose python is not thread safe and the rospy client side library
        # uses a thread per pub/sub and service. Since the openpose wrapper is created in the main thread, we have
        # to communicate our openpose requests (inputs) to the main thread where the request is processed by the
        # openpose wrapper (step 1).
        # We have a separate spin loop in the main thead that checks whether there are items in the input q and
        # processes these using the Openpose wrapper (step 2).
        # When the processing has finished, we add the result in the corresponding output queue (specified by the
        # request in the input queue) (step 3).
        self._input_q = Queue()  # image_msg, save_images, publish_images, is_service_request
        self._service_output_q = Queue()  # recognitions
        self._subscriber_output_q = Queue()  # recognitions

        # Debug
        self._topic_save_images = topic_save_images
        self._service_save_images = service_save_images
        self._topic_publish_result_image = topic_publish_result_image
        self._service_publish_result_image = service_publish_result_image
        self._save_images_folder = save_images_folder

        # ROS IO
        self._bridge = CvBridge()
        self._recognize_srv = rospy.Service("recognize", Recognize, self._recognize_srv)
        self._image_subscriber = rospy.Subscriber("image", Image, self._image_callback)
        self._recognitions_publisher = rospy.Publisher("recognitions", Recognitions, queue_size=10)
        if self._topic_publish_result_image or self._service_publish_result_image:
            self._result_image_publisher = rospy.Publisher("result_image", Image, queue_size=10)

        self.last_master_check = rospy.get_time()

        rospy.loginfo("PoseEstimationNode initialized:")
        rospy.loginfo(f" - {model_name=}")
        rospy.loginfo(f" - {device=}")
        rospy.loginfo(f" - {conf=}")
        rospy.loginfo(f" - {topic_save_images=}")
        rospy.loginfo(f" - {service_save_images=}")
        rospy.loginfo(f" - {topic_publish_result_image=}")
        rospy.loginfo(f" - {service_publish_result_image=}")
        rospy.loginfo(f" - {save_images_folder=}")

    def _image_callback(self, image_msg):
        self._input_q.put((image_msg, self._topic_save_images, self._topic_publish_result_image, False))
        self._recognitions_publisher.publish(
            Recognitions(header=image_msg.header, recognitions=self._subscriber_output_q.get())
        )

    def _recognize_srv(self, req):
        self._input_q.put((req.image, self._service_save_images, self._service_publish_result_image, True))
        return {"recognitions": self._service_output_q.get()}

    def _get_recognitions(self, image_msg, save_images, publish_images):
        """
        Handles the recognition and publishes and stores the debug images (should be called in the main thread)

        :param image_msg: Incoming image
        :param save_images: Whether to store the images
        :param publish_images: Whether to publish the images
        :return: The recognized recognitions
        """
        try:
            bgr_image = self._bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert to opencv image: {e}")
            return []

        recognitions, result_image = self._wrapper.detect_poses(bgr_image)

        # Write images
        if save_images:
            image_writer.write_raw(self._save_images_folder, bgr_image)
            image_writer.write_raw(self._save_images_folder, result_image, "overlayed")

        if publish_images:
            self._result_image_publisher.publish(self._bridge.cv2_to_imgmsg(result_image, "bgr8"))

        # Service response
        return recognitions

    def spin(self, check_master: bool = False):
        """
        Empty input queues and fill output queues (see __init__ doc)
        """
        while not rospy.is_shutdown():
            try:
                image_msg, save_images, publish_images, is_service_request = self._input_q.get(timeout=1.0)
            except Empty:
                pass
            else:
                if is_service_request:
                    self._service_output_q.put(self._get_recognitions(image_msg, save_images, publish_images))
                else:
                    self._subscriber_output_q.put(self._get_recognitions(image_msg, save_images, publish_images))
            finally:
                if check_master and rospy.get_time() >= self.last_master_check + 1:
                    self.last_master_check = rospy.get_time()
                    try:
                        rospy.get_master().getPid()
                    except socket.error:
                        rospy.logdebug("Connection to master is lost")
                        return 1  # This should result in a non-zero error code of the entire program

        return 0


if __name__ == "__main__":
    rospy.init_node("pose_estimation")

    try:
        node = PoseEstimationNode(
            rospy.get_param("~model", "yolov8n-pose.pt"),
            rospy.get_param("~device", "cuda:0"),
            rospy.get_param("~conf", 0.25),
            rospy.get_param("~topic_save_images", False),
            rospy.get_param("~service_save_images", True),
            rospy.get_param("~topic_publish_result_image", True),
            rospy.get_param("~service_publish_result_image", True),
            os.path.expanduser(rospy.get_param("~save_images_folder", "/tmp/pose_estimation")),
        )

        check_master: bool = rospy.get_param("~check_master", False)

        updater = diagnostic_updater.Updater()
        updater.setHardwareID("none")
        updater.add(diagnostic_updater.Heartbeat())
        rospy.Timer(rospy.Duration(1), lambda event: updater.force_update())

        sys.exit(node.spin(check_master))
    except Exception as e:
        rospy.logfatal(e)
        raise
