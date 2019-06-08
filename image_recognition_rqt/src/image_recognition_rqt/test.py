import rospy
import rosservice
import rostopic
from cv_bridge import CvBridge, CvBridgeError
from dialogs import option_dialog, warning_dialog, info_dialog
from image_recognition_msgs.msg import CategoryProbability, FaceProperties
from image_recognition_msgs.srv import GetFaceProperties, Recognize
from image_widget import ImageWidget
from python_qt_binding.QtCore import *
from python_qt_binding.QtGui import *
from python_qt_binding.QtWidgets import *
from qt_gui.plugin import Plugin
from sensor_msgs.msg import Image

_SUPPORTED_SERVICES = ["image_recognition_msgs/Recognize",
                       "image_recognition_msgs/GetFaceProperties"]


class TestPlugin(Plugin):

    def __init__(self, context):
        """
        TestPlugin class to evaluate the image_recognition_msgs interfaces
        :param context: QT context, aka parent
        """
        super(TestPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Test Plugin')

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

        self._info = QLineEdit()
        self._info.setDisabled(True)
        self._info.setText("Draw a rectangle on the screen to perform recognition of that ROI")
        layout.addWidget(self._info)

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        # Set subscriber and service to None
        self._sub = None
        self._srv = None

    def recognize_srv_call(self, roi_image):
        """
        Method that calls the Recognize.srv
        :param roi_image: Selected roi_image by the user
        """
        try:
            result = self._srv(image=self.bridge.cv2_to_imgmsg(roi_image, "bgr8"))
        except Exception as e:
            warning_dialog("Service Exception", str(e))
            return

        for r in result.recognitions:
            text_array = []
            best = CategoryProbability(label="unknown", probability=r.categorical_distribution.unknown_probability)
            for p in r.categorical_distribution.probabilities:
                text_array.append("%s: %.2f" % (p.label, p.probability))
                if p.probability > best.probability:
                    best = p

            self._image_widget.add_detection(r.roi.x_offset, r.roi.y_offset, r.roi.width, r.roi.height, best.label)

            if text_array:
                option_dialog("Classification results (Unknown probability=%.2f)" %
                              r.categorical_distribution.unknown_probability,
                              text_array)  # Show all results in a dropdown

    def get_face_properties_srv_call(self, roi_image):
        """
        Method that calls the GetFaceProperties.srv
        :param roi_image: Selected roi_image by the user
        """
        try:
            result = self._srv(face_image_array=[self.bridge.cv2_to_imgmsg(roi_image, "bgr8")])
        except Exception as e:
            warning_dialog("Service Exception", str(e))
            return

        msg = ""
        for properties in result.properties_array:
            msg += "- FaceProperties(gender=%s, gender_confidence=%.2f, age=%s)" % \
                   ("male" if properties.gender == FaceProperties.MALE else "female", properties.gender_confidence,
                    properties.age)

        info_dialog("Face Properties array", msg)

    def image_roi_callback(self, roi_image):
        """
        Callback triggered when the user has drawn an ROI on the image
        :param roi_image: The opencv image in the ROI
        """
        if self._srv is None:
            warning_dialog("No service specified!",
                           "Please first specify a service via the options button (top-right gear wheel)")
            return

        if self._srv.service_class == Recognize:
            self.recognize_srv_call(roi_image)
        elif self._srv.service_class == GetFaceProperties:
            self.get_face_properties_srv_call(roi_image)
        else:
            warning_dialog("Unknown service class", "Service class is unkown!")

    def _image_callback(self, msg):
        """
        Sensor_msgs/Image callback
        :param msg: The image message
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        self._image_widget.set_image(cv_image)

    def trigger_configuration(self):
        """
        Callback when the configuration button is clicked
        """
        topic_name, ok = QInputDialog.getItem(self._widget, "Select topic name", "Topic name",
                                              rostopic.find_by_type('sensor_msgs/Image'))
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
        """
        Method that creates a subscriber to a sensor_msgs/Image topic
        :param topic_name: The topic_name
        """
        if self._sub:
            self._sub.unregister()
        self._sub = rospy.Subscriber(topic_name, Image, self._image_callback)
        rospy.loginfo("Listening to %s -- spinning .." % self._sub.name)
        self._widget.setWindowTitle("Test plugin, listening to (%s)" % self._sub.name)

    def _create_service_client(self, srv_name):
        """
        Method that creates a client service proxy to call either the GetFaceProperties.srv or the Recognize.srv
        :param srv_name:
        """
        if self._srv:
            self._srv.close()

        if srv_name in rosservice.get_service_list():
            rospy.loginfo("Creating proxy for service '%s'" % srv_name)
            self._srv = rospy.ServiceProxy(srv_name, rosservice.get_service_class_by_name(srv_name))

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
        if self._sub:
            instance_settings.set_value("topic_name", self._sub.name)

    def restore_settings(self, plugin_settings, instance_settings):
        """
        Callback function fired on load of the plugin that allows to restore saved variables
        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
        self._create_subscriber(str(instance_settings.value("topic_name", "/usb_cam/image_raw")))
        self._create_service_client(str(instance_settings.value("service_name", "/image_recognition/my_service")))
