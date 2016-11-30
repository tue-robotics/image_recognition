import rospy
import rostopic
import rosservice

from qt_gui.plugin import Plugin

from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 

from sensor_msgs.msg import RegionOfInterest
from cv_bridge import CvBridge, CvBridgeError
from image_recognition_msgs.msg import CategoryProbability, Recognition

from image_widget import ImageWidget
from dialogs import option_dialog, warning_dialog

from image_recognition_msgs.srv import Recognize, RecognizeResponse
import re


def _sanitize(label):
    """
    Sanitize string, only allow \w regex chars
    :param label: Input that needs to be sanitized
    :return: The sanatized string
    """
    return re.sub(r'(\W+| )', '', label)


class ManualPlugin(Plugin):

    def __init__(self, context):
        """
        ManualPlugin class that performs a manual recognition based on a request
        :param context: QT context, aka parent
        """
        super(ManualPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Manual Plugin')

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

        self._labels_edit = QLineEdit()
        self._labels_edit.setDisabled(True)
        grid_layout.addWidget(self._labels_edit, 2, 2)

        self._edit_labels_button = QPushButton("Edit labels")
        self._edit_labels_button.clicked.connect(self._get_labels)
        grid_layout.addWidget(self._edit_labels_button, 2, 1)

        self._done_recognizing_button = QPushButton("Done recognizing..")
        self._done_recognizing_button.clicked.connect(self._done_recognizing)
        self._done_recognizing_button.setDisabled(True)
        grid_layout.addWidget(self._done_recognizing_button, 3, 2)

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        # Set service to None
        self._srv = None
        self._srv_name = None

        self._response = RecognizeResponse()
        self._recognizing = False

    def _get_labels(self):
        """
        Gets and sets the labels
        """
        text, ok = QInputDialog.getText(self._widget, 'Text Input Dialog',
                                        'Type labels semicolon separated, e.g. banana;apple:',
                                        QLineEdit.Normal, ";".join(self.labels))
        if ok:
            # Sanitize to alphanumeric, exclude spaces
            labels = set([_sanitize(label) for label in str(text).split(";") if _sanitize(label)])
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

    def _done_recognizing(self):
        self._image_widget.clear()
        self._recognizing = False

    def recognize_srv_callback(self, req):
        """
        Method callback for the Recognize.srv
        :param req: The service request
        """
        self._response.recognitions = []
        self._recognizing = True

        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        self._image_widget.set_image(cv_image)
        self._done_recognizing_button.setDisabled(False)

        timeout = 60.0  # Maximum of 60 seconds
        future = rospy.Time.now() + rospy.Duration(timeout)
        rospy.loginfo("Waiting for manual recognition, maximum of %d seconds", timeout)
        while not rospy.is_shutdown() and self._recognizing:
            if rospy.Time.now() > future:
                raise rospy.ServiceException("Timeout of %d seconds exceeded .." % timeout)
            rospy.sleep(rospy.Duration(0.1))

        self._done_recognizing_button.setDisabled(True)

        return self._response

    def image_roi_callback(self, roi_image):
        """
        Callback triggered when the user has drawn an ROI on the image
        :param roi_image: The opencv image in the ROI
        """
        if not self.labels:
            warning_dialog("No labels specified!", "Please first specify some labels using the 'Edit labels' button")
            return

        height, width = roi_image.shape[:2]

        option = option_dialog("Label", self.labels)
        if option:
            self._image_widget.add_detection(0, 0, width, height, option)
            self._stage_recognition(self._image_widget.get_roi(), option)

    def _stage_recognition(self, roi, label):
        """
        Stage a manual recognition
        :param roi: ROI
        :param label: The label
        """
        x, y, width, height = roi
        r = Recognition(roi=RegionOfInterest(x_offset=x, y_offset=y, width=width, height=height))
        r.categorical_distribution.probabilities = [CategoryProbability(label=label, probability=1.0)]
        r.categorical_distribution.unknown_probability = 0.0

        self._response.recognitions.append(r)

    def trigger_configuration(self):
        """
        Callback when the configuration button is clicked
        """

        srv_name, ok = QInputDialog.getText(self._widget, "Select service name", "Service name")
        if ok:
            self._create_service_server(srv_name)

    def _create_service_server(self, srv_name):
        """
        Method that creates a service server for a Recognize.srv
        :param srv_name:
        """
        if self._srv:
            self._srv.shutdown()

        if srv_name:
            rospy.loginfo("Creating service '%s'" % srv_name)
            self._srv_name = srv_name
            self._srv = rospy.Service(srv_name, Recognize, self.recognize_srv_callback)

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
        instance_settings.set_value("labels", self.labels)
        if self._srv:
            instance_settings.set_value("srv_name", self._srv_name)

    def restore_settings(self, plugin_settings, instance_settings):
        """
        Callback function fired on load of the plugin that allows to restore saved variables
        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
        labels = None
        try:
            labels = instance_settings.value("labels")
        except:
            pass
        self._set_labels(labels)
        self._create_service_server(str(instance_settings.value("srv_name", "/my_recognition_service")))
