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

def _warn_msg(title, text):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(text)
    msg.setWindowTitle(title)
    msg.exec_()

def _sanitize(label):
    return re.sub(r'(\W+| )', '', label)

def _write_image_to_file(path, image, x, y, width, height, label):
    # Check if path exists
    if not os.path.exists(path):
        rospy.logerr("Path %s does not exist", path)
        return

    # Check if path label exist, otherwise created it
    label_folder = path + "/" + label
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

    filename = "%s/%s.jpg" % (label_folder, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    cv2.imwrite(filename, image[y: y + height, x: x + width])

    rospy.loginfo("Wrote file to %s", filename)

def _convert_cv_to_qt_image(cv_image):
    cv_image = cv_image.copy() # Create a copy
    height, width, byte_value = cv_image.shape
    byte_value = byte_value * width
    cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB, cv_image)

    return QImage(cv_image, width, height, byte_value, QImage.Format_RGB888)

class LabelDialog(QDialog):
    def __init__(self, labels):
        super(LabelDialog, self).__init__()

        self.layout = QGridLayout()
        self.label_selector = QComboBox()
        self.label_selector.addItems(labels)
        self.layout.addWidget(QLabel("Label:"), 0, 0)
        self.layout.addWidget(self.label_selector, 0, 1)
        self.ok_button = QPushButton('ok')
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button, 0, 2)

        self.setLayout(self.layout)

    def keyPressEvent(self, event):
        super(LabelDialog, self).keyPressEvent(event)
        if event.key() == Qt.Key_Return:
            self.accept()

class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)   
        self._cv_image = None
        self._cv_mask = None 
        self._qt_image = QImage()

        self.largest_rect = QRect(50, 50, 400, 400)
        
        self.clip_rect = QRect(0,0,0,0)
        self.dragging = False
        self.drag_offset = QPoint()
        self.labels = []
        self.label = ""

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self._qt_image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.cyan, 5.0))
        painter.drawRect(self.clip_rect)
        painter.setFont(QFont('Decorative', 10))
        painter.drawText(self.clip_rect, Qt.AlignCenter, self.label)   
        painter.end()

    def set_image(self, image):
        self._cv_image = image

        # Mask image if mask exists
        if self._cv_mask is not None:
            image *= self._cv_mask

        self._qt_image = _convert_cv_to_qt_image(image)
        self.update()

    def mousePressEvent(self, event):
        # Check if we clicked on the img
        if event.pos().x() < self._qt_image.width() and event.pos().y() < self._qt_image.height():
            self.clip_rect.setTopLeft(event.pos())
            self.dragging = True
    
    def mouseMoveEvent(self, event):
        if not self.dragging:
            return

        self.clip_rect.setBottomRight(event.pos())
        
        self.update()
    
    def mouseReleaseEvent(self, event):
        if not self.dragging:
            return

        if not self.labels:
            _warn_msg("No labels specified!", "Please first specify some labels using the 'Edit labels' button")
            return

        dlg = LabelDialog(self.labels)
        if dlg.exec_():
            self.label = str(dlg.label_selector.currentText())

        self.store_image()

        self.dragging = False

    def store_image(self):
        if not self.label:
            _warn_msg("No label specified!", "Please specify a label first")
        else:
            _write_image_to_file(self.output_directory, self._cv_image, self.clip_rect.x(), 
                self.clip_rect.y(), self.clip_rect.width(), self.clip_rect.height(), self.label)

class LabelPlugin(Plugin):

    def __init__(self, context):
        super(LabelPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Label Plugin')

        self._widget = QWidget()
        context.add_widget(self._widget)

        # add key handler
        #self._widget.keyPressed.connect(self._key_pressed)
        
        # Layout and attach to widget
        layout = QVBoxLayout()  
        self._widget.setLayout(layout)

        self._image_widget = ImageWidget()
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
        self._save_button.clicked.connect(self._image_widget.store_image)
        grid_layout.addWidget(self._save_button, 2, 3)

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        # Set subscriber to None
        self._sub = None

    def _get_output_directory(self):
        self._set_output_directory(QFileDialog.getExistingDirectory(self._widget, "Select output directory"))

    def _set_output_directory(self, path):
        if not path:
            path = "/tmp"

        self._image_widget.output_directory = path
        self._output_path_edit.setText("Saving images to %s" % path)

    def _get_labels(self):
        text, ok = QInputDialog.getText(self._widget, 'Text Input Dialog', 'Type labels semicolon separated, e.g. banana;apple:')
        if ok:
            labels = set([_sanitize(label) for label in str(text).split(";") if _sanitize(label)]) # Sanitize to alphanumeric, exclude spaces
            self._set_labels(labels)

    def _set_labels(self, labels):
        if not labels:
            labels = []

        self._image_widget.labels = labels
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
        instance_settings.set_value("output_directory", self._image_widget.output_directory)
        instance_settings.set_value("labels", self._image_widget.labels)
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

    def _key_pressed(self, event):
        self._image_widget.store_image()