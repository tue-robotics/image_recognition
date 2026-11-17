from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_service
import os

from qt_gui.plugin import Plugin

from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import re

from .image_widget import ImageWidget
from .dialogs import option_dialog, warning_dialog

from image_recognition_interfaces.msg import Annotation
from image_recognition_interfaces.srv import Annotate
from image_recognition_util import image_writer
from sensor_msgs.msg import RegionOfInterest

_SUPPORTED_SERVICES = ["image_recognition_interfaces/srv/Annotate"]


def _sanitize(label):
    r"""
    Sanitize string, only allow \w regex chars

    :param label: Input that needs to be sanitized
    :return: The sanatized string
    """
    return re.sub(r'(\W+| )', '', label)


class AnnotationPlugin(Plugin):
    _image_received = Signal(object)
    _service_finished = Signal(object, object)

    def __init__(self, context):
        """
        Annotation plugin to create data sets or test the Annotate.srv service

        :param context: Parent QT widget
        """
        super(AnnotationPlugin, self).__init__(context)
        self._node = context.node

        # Widget setup
        self.setObjectName('Label Plugin')

        self._widget = QWidget()
        context.add_widget(self._widget)
        
        # Layout and attach to widget
        layout = QVBoxLayout()  
        self._widget.setLayout(layout)

        self._image_widget = ImageWidget(self._widget, self.image_roi_callback, clear_on_click=True)
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

        self._save_button = QPushButton("Annotate again!")
        self._save_button.clicked.connect(self.annotate_again_clicked)
        grid_layout.addWidget(self._save_button, 2, 3)

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        # Set subscriber to None
        self._sub = None
        self._srv = None
        self._image_received.connect(self._image_callback, Qt.QueuedConnection)
        self._service_finished.connect(self._handle_service_result, Qt.QueuedConnection)

        self.labels = []
        self.label = ""
        self.output_directory = ""

    def image_roi_callback(self, roi_image):
        """
        Callback from the image widget when the user has selected a ROI

        :param roi_image: The opencv image of the ROI
        """
        if not self.labels:
            warning_dialog("No labels specified!", "Please first specify some labels using the 'Edit labels' button")
            return

        height, width = roi_image.shape[:2]

        option = option_dialog("Label", self.labels)
        if option:
            self.label = option
            self._image_widget.add_detection(0, 0, width, height, option)
            self.annotate(roi_image)

    def annotate_again_clicked(self):
        """
        Triggered when button clicked
        """
        roi_image = self._image_widget.get_roi_image()
        if roi_image is not None:
            self.annotate(roi_image)

    def annotate(self, roi_image):
        """
        Create an annotation

        :param roi_image: The image we want to annotate
        """
        self.annotate_srv(roi_image)
        self.store_image(roi_image)

    def annotate_srv(self, roi_image):
        """
        Call the selected Annotate.srv
        :param roi_image: The full opencv image we want to annotate
        """
        if roi_image is not None and self.label is not None and self._srv is not None:
            height, width = roi_image.shape[:2]
            try:
                self._call_service(Annotate.Request(image=self.bridge.cv2_to_imgmsg(roi_image, "bgr8"),
                          annotations=[Annotation(label=self.label,
                                                  roi=RegionOfInterest(x_offset=0, y_offset=0,
                                                                       width=width, height=height))]))
            except Exception as e:
                warning_dialog("Service Exception", str(e))

    def _create_service_client(self, srv_name):
        """
        Create a service client proxy

        :param srv_name: Name of the service
        """
        if self._srv and srv_name == self._srv.srv_name:
            return
        service_types = dict(self._node.get_service_names_and_types()).get(srv_name, [])
        service_type = next((name for name in service_types if name in _SUPPORTED_SERVICES), None)
        if service_type is None:
            return
        if self._srv:
            self._node.destroy_client(self._srv)
        self._srv = self._node.create_client(get_service(service_type), srv_name)

    def _call_service(self, request, on_result=None):
        if not self._srv.service_is_ready():
            warning_dialog("Service unavailable", self._srv.srv_name)
            return
        future = self._srv.call_async(request)
        future.add_done_callback(lambda result: self._service_finished.emit(result, on_result))

    def _handle_service_result(self, future, on_result):
        try:
            result = future.result()
        except Exception as error:
            warning_dialog("Service Exception", str(error))
            return
        if on_result is not None:
            on_result(result)


    def store_image(self, roi_image):
        """
        Store the image

        :param roi_image: Image we would like to store
        """
        if roi_image is not None and self.label is not None and self.output_directory is not None:
            image_writer.write_annotated(self.output_directory, roi_image, self.label, True)

    def _get_output_directory(self):
        """
        Gets and sets the output directory via a QFileDialog
        """
        self._set_output_directory(QFileDialog.getExistingDirectory(self._widget, "Select output directory"))

    def _set_output_directory(self, path):
        """
        Sets the output directory

        :param path: The path of the directory
        """
        if not path:
            path = "/tmp"

        self.output_directory = path
        self._output_path_edit.setText("Saving images to %s" % path)

    def _get_labels(self):
        """
        Gets and sets the labels
        """
        text, ok = QInputDialog.getText(self._widget, 'Text Input Dialog', 'Type labels semicolon separated, e.g. banana;apple:',
            QLineEdit.Normal, ";".join(self.labels))
        if ok:
            labels = set([_sanitize(label) for label in str(text).split(";") if _sanitize(label)]) # Sanitize to alphanumeric, exclude spaces
            self._set_labels(labels)

    def _set_labels(self, labels):
        """
        Sets the labels
        :param labels: label string array
        """
        if not labels:
            labels = []

        self.labels = labels
        self._labels_edit.setText("%s" % labels)

    def _image_callback(self, msg):
        """
        Called when a new sensor_msgs/Image is coming in
        :param msg: The image messaeg
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self._node.get_logger().error(str(e))
            return

        self._image_widget.set_image(cv_image)

    def trigger_configuration(self):
        """
        Callback when the configuration button is clicked
        """
        image_topics = sorted(name for name, types in self._node.get_topic_names_and_types() if 'sensor_msgs/msg/Image' in types)
        try:
            topic_index = image_topics.index(self._sub.topic_name)
        except (AttributeError, ValueError):
            topic_index = 0

        topic_name, ok = QInputDialog.getItem(self._widget, "Select topic name", "Topic name",
                                              image_topics, topic_index)
        if ok:
            self._create_subscriber(topic_name)

        available_rosservices = sorted(
            name for name, types in self._node.get_service_names_and_types()
            if any(service_type in _SUPPORTED_SERVICES for service_type in types)
        )

        try:
            srv_index = available_rosservices.index(self._srv.srv_name)
        except (AttributeError, ValueError):
            srv_index = 0

        srv_name, ok = QInputDialog.getItem(self._widget, "Select service name", "Service name", available_rosservices,
                                            srv_index)
        if ok:
            self._create_service_client(srv_name)

    def _create_subscriber(self, topic_name):
        """
        Method that creates a subscriber to a sensor_msgs/Image topic
        :param topic_name: The topic_name
        """
        if self._sub:
            if topic_name == self._sub.topic_name:
                return
            self._node.destroy_subscription(self._sub)

        self._sub = self._node.create_subscription(Image, topic_name, lambda image: self._image_received.emit(image), qos_profile_sensor_data)
        self._widget.setWindowTitle("Label plugin, listening to (%s)" % self._sub.topic_name)

    def shutdown_plugin(self):
        """
        Callback function when shutdown is requested
        """
        if self._sub:
            self._node.destroy_subscription(self._sub)
        if self._srv:
            self._node.destroy_client(self._srv)

    def save_settings(self, plugin_settings, instance_settings):
        """
        Callback function on shutdown to store the local plugin variables

        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
        instance_settings.set_value("output_directory", self.output_directory)
        instance_settings.set_value("labels", self.labels)
        if self._sub:
            instance_settings.set_value("topic_name", self._sub.topic_name)
        if self._srv:
            instance_settings.set_value("service_name", self._srv.srv_name)

    def restore_settings(self, plugin_settings, instance_settings):
        """
        Callback function fired on load of the plugin that allows to restore saved variables

        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
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

        self._create_subscriber(str(instance_settings.value("topic_name", "/usb_cam/image_raw")))
        self._create_service_client(str(instance_settings.value("service_name", "/image_recognition/my_service")))