import rospy
from functools import partial
import os
import fnmatch
import collections

from qt_gui.plugin import Plugin

from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def find_files(directory, patterns):
    for pattern in patterns:
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename


class FolderImagePublisherPlugin(Plugin):

    def __init__(self, context):
        """
        FolderImagePublisherPlugin class that publishes images from a folder
        :param context: QT context, aka parent
        """
        super(FolderImagePublisherPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('FolderImagePublisherPlugin')

        self._widget = QWidget()
        context.add_widget(self._widget)
        
        # Layout and attach to widget
        layout = QHBoxLayout()
        self._widget.setLayout(layout)

        self._info = QLineEdit()
        self._info.setDisabled(True)
        self._info.setText("Please choose a directory (top-right corner)")
        layout.addWidget(self._info)

        self._left_button = QPushButton('<')
        self._left_button.clicked.connect(partial(self._rotate_and_publish, -1))
        layout.addWidget(self._left_button)

        self._right_button = QPushButton('>')
        self._right_button.clicked.connect(partial(self._rotate_and_publish, 1))
        layout.addWidget(self._right_button)

        # Set subscriber and service to None
        self._pub = rospy.Publisher("folder_image", Image, queue_size=1)

        self._bridge = CvBridge()

        self._files = collections.deque([])

    def trigger_configuration(self):
        """
        Callback when the configuration button is clicked
        """
        self._index_image_files(QFileDialog.getExistingDirectory(self._widget, "Select image directory"))

    def _index_image_files(self, directory):
        """
        Index all images in the specified directory
        """
        self._files = collections.deque([f for f in find_files(directory, ['*.jpeg', '*.jpg', '*.png'])])
        self._publish_image()

    def _publish_image(self):
        """
        Publish the first image from the deque
        """
        if self._files:
            self._info.setText(self._files[0])
            self._pub.publish(self._bridge.cv2_to_imgmsg(cv2.imread(self._files[0]), encoding="bgr8"))

    def _rotate_and_publish(self, rotate_arg):
        self._files.rotate(rotate_arg)
        self._publish_image()

    def shutdown_plugin(self):
        """
        Callback function when shutdown is requested
        """
        pass

    def save_settings(self, plugin_settings, instance_settings):
        """
        Callback function on shutdown to store the local plugin variables
        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        """
        Callback function fired on load of the plugin that allows to restore saved variables
        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
        pass
