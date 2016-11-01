import os
import rospy
import rospkg
import rostopic
import subprocess
import rosservice

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi

from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import datetime
import re

from image_widget import ImageWidget
from dialogs import option_dialog, warning_dialog

_SUPPORTED_SERVICES = ["image_recognition_msgs/Recognize", "image_recognition_msgs/GetFaceProperties"]


class TestPlugin(Plugin):

    def __init__(self, context):
        super(TestPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Test Plugin')

        self._widget = QWidget()
        context.add_widget(self._widget)
        
        # Layout and attach to widget
        layout = QVBoxLayout()  
        self._widget.setLayout(layout)

        self._image_widget = ImageWidget(self._widget, self.image_roi_callback)
        layout.addWidget(self._image_widget)

        # Input field
        grid_layout = QGridLayout()
        layout.addLayout(grid_layout)

        self._info = QLineEdit()
        self._info.setDisabled(True)
        self._info.setText("Draw a rectangle on the screen to perform recognition of that ROI")
        layout.addWidget(self._info)

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        # Set subscriber and service to None
        self._sub = None
        self._srv = None

    def image_roi_callback(self, roi_image):
        if self._srv is None:
            warning_dialog("No service specified!", "Please first specify a service via the options button (top-right gear wheel)")
            return

        try:
            result = self._srv(image=self.bridge.cv2_to_imgmsg(roi_image, "bgr8"))
        except Exception as e:
            warning_dialog("Service Exception", str(e))
            return

        for recognition in result.recognitions: # TODO Draw results of multiple recognitions correctly and deal with unknown
            text_array = [ "%s: %.2f" % (p.label, p.probability) for p in recognition.categorical_distribution.probabilities ]

            if text_array:
                self._image_widget.set_text(text_array[0]) # Show first option in the image
                option_dialog("Classification results", text_array) # Show all results in a dropdown

    def _image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        self._image_widget.set_image(cv_image)

    def trigger_configuration(self):
        topic_name, ok = QInputDialog.getItem(self._widget, "Select topic name", "Topic name", rostopic.find_by_type('sensor_msgs/Image'))
        if ok:
            self._create_subscriber(topic_name)

        available_rosservices = []
        for s in rosservice.get_service_list():
            try:
                if rosservice.get_service_type(s) in _SUPPORTED_SERVICES:
                    available_rosservices.append(s)
            except:
                pass

        srv_name, ok = QInputDialog.getItem(self._widget, "Select service name", "Service name", available_rosservices)
        if ok:
            self._create_service_client(srv_name)

    def _create_subscriber(self, topic_name):
        if self._sub:
            self._sub.unregister()
        self._sub = rospy.Subscriber(topic_name, Image, self._image_callback)
        rospy.loginfo("Listening to %s -- spinning .." % self._sub.name)
        self._widget.setWindowTitle("Test plugin, listening to (%s)" % self._sub.name)

    def _create_service_client(self, srv_name):
        if self._srv:
            self._srv.close()

        if srv_name in rosservice.get_service_list():
            rospy.loginfo("Creating proxy for service '%s'" % srv_name)
            self._srv = rospy.ServiceProxy(srv_name, rosservice.get_service_class_by_name(srv_name))

    def shutdown_plugin(self):
        pass

    def save_settings(self, plugin_settings, instance_settings):
        if self._sub:
            instance_settings.set_value("topic_name", self._sub.name)

    def restore_settings(self, plugin_settings, instance_settings):
        self._create_subscriber(str(instance_settings.value("topic_name", "/usb_cam/image_raw")))
        self._create_service_client(str(instance_settings.value("service_name", "/image_recognition/my_service")))