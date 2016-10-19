import os
import rospy
import rospkg
import rostopic
import subprocess

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

def _sanitize(label):
    return re.sub(r'(\W+| )', '', label)

def _write_image_to_file(path, image, label):
    # Check if path exists
    if not os.path.exists(path):
        rospy.logerr("Path %s does not exist", path)
        return

    # Check if path label exist, otherwise created it
    label_folder = path + "/" + label
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    filename = "%s/%s.jpg" % (label_folder, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    cv2.imwrite(filename, image)

    rospy.loginfo("Wrote file to %s", filename)

class LabelPlugin(Plugin):

    def __init__(self, context):
        super(LabelPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Label Plugin')

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

        self._edit_path_button = QPushButton("Edit path")
        self._edit_path_button.clicked.connect(self._get_output_directory)
        grid_layout.addWidget(self._edit_path_button, 1, 1)

        self._output_path_edit = QLineEdit()
        self._output_path_edit.setDisabled(True)
        grid_layout.addWidget(self._output_path_edit, 1, 2)

        self._labels_edit = QLineEdit()
        self._labels_edit.setDisabled(True)
        grid_layout.addWidget(self._labels_edit, 2, 2)

        self._edit_labels_button = QPushButton("Edit labels")
        self._edit_labels_button.clicked.connect(self._get_labels)
        grid_layout.addWidget(self._edit_labels_button, 2, 1)

        self._save_button = QPushButton("Save another one")
        self._save_button.clicked.connect(self.store_image)
        grid_layout.addWidget(self._save_button, 2, 3)

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        # Set subscriber to None
        self._sub = None

        self.labels = []
        self.roi_image = None
        self.label = ""
        self.output_directory = ""

    def image_roi_callback(self, roi_image):
        if not self.labels:
            warning_dialog("No labels specified!", "Please first specify some labels using the 'Edit labels' button")
            return

        self.roi_image = roi_image

        option = option_dialog("Label", self.labels)
        if option:
            self.label = option
            self._image_widget.set_text(option)

        self.store_image()
            
    def store_image(self):
        if not None in [self.roi_image, self.label, self.output_directory]:
            _write_image_to_file(self.output_directory, self.roi_image, self.label)

    def _get_output_directory(self):
        self._set_output_directory(QFileDialog.getExistingDirectory(self._widget, "Select output directory"))

    def _set_output_directory(self, path):
        if not path:
            path = "/tmp"

        self.output_directory = path
        self._output_path_edit.setText("Saving images to %s" % path)

    def _get_labels(self):
        text, ok = QInputDialog.getText(self._widget, 'Text Input Dialog', 'Type labels semicolon separated, e.g. banana;apple:', 
            QLineEdit.Normal, ";".join(self.labels))
        if ok:
            labels = set([_sanitize(label) for label in str(text).split(";") if _sanitize(label)]) # Sanitize to alphanumeric, exclude spaces
            self._set_labels(labels)

    def _set_labels(self, labels):
        if not labels:
            labels = []

        self.labels = labels
        self._labels_edit.setText("%s" % labels)

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

    def _create_subscriber(self, topic_name):
        if self._sub:
            self._sub.unregister()
        self._sub = rospy.Subscriber(topic_name, Image, self._image_callback)
        rospy.loginfo("Listening to %s -- spinning .." % self._sub.name)
        self._widget.setWindowTitle("Label plugin, listening to (%s)" % self._sub.name)

    def shutdown_plugin(self):
        pass

    def save_settings(self, plugin_settings, instance_settings):
        instance_settings.set_value("output_directory", self.output_directory)
        instance_settings.set_value("labels", self.labels)
        if self._sub:
            instance_settings.set_value("topic_name", self._sub.name)

    def restore_settings(self, plugin_settings, instance_settings):
        path = None
        try:
            path = instance_settings.value("output_directory")
        except:
            pass
        self._set_output_directory(path)

        labels = None
        try:
            labels = instance_settings.value("labels")
        except:
            pass
        self._set_labels(labels)

        self._create_subscriber(str(instance_settings.value("topic_name","/usb_cam/image_raw")))