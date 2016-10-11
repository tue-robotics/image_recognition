import os
import rospy
import rospkg
import subprocess

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding import QtGui , QtCore, QtWidgets
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import datetime

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
    def __init__(self):
        super(LabelDialog, self).__init__()

        self.layout = QGridLayout()
        self.label_selector = QComboBox()
        self.label_selector.addItems(["omg", "blaat"])
        self.layout.addWidget(QLabel("Label:"), 0, 0)
        self.layout.addWidget(self.label_selector, 0, 1)
        self.ok_button = QPushButton('ok')
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button, 0, 2)

        self.setLayout(self.layout)
        self.output_directory = "/tmp"

    def keyPressEvent(self, event):
        super(LabelDialog, self).keyPressEvent(event)
        if event.key() == Qt.Key_Return:
            self.accept()

class ImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)   
        self._cv_image = None
        self._cv_mask = None 
        self._qt_image = QImage()

        self.largest_rect = QRect(50, 50, 400, 400)
        
        self.clip_rect = QRect(0,0,0,0)
        self.dragging = False
        self.drag_offset = QPoint()

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self._qt_image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.cyan, 5.0))
        painter.drawRect(self.clip_rect)
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

        dlg = LabelDialog()
        if dlg.exec_():
            _write_image_to_file(self.output_directory, self._cv_image, self.clip_rect.x(), self.clip_rect.y(), 
                self.clip_rect.width(), self.clip_rect.height(), str(dlg.label_selector.currentText()))

        self.dragging = False


class LabelPlugin(Plugin):

    def __init__(self, context):
        super(LabelPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Record Plugin')

        self._widget = QtWidgets.QWidget()
        context.add_widget(self._widget)
        
        # Layout and attach to widget
        layout = QtWidgets.QVBoxLayout()  
        self._widget.setLayout(layout)

        self._image_widget = ImageWidget()
        layout.addWidget(self._image_widget)

        # Input field
        self._edit = QtWidgets.QLineEdit()
        self._edit.mousePressEvent = lambda _ : self._get_output_directory()

        layout.addWidget(self._edit)

        # Record button
        self._set_output_directory("/tmp")

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        # Setup listener
        self._sub = rospy.Subscriber("/usb_cam/image_raw", Image, self._image_callback)
        rospy.loginfo("Listening to %s -- spinning .." % self._sub.name)

    def _get_output_directory(self):
        self._set_output_directory(QtWidgets.QFileDialog.getExistingDirectory(self._widget, "Select output directory"))

    def _set_output_directory(self, path):
        self._image_widget.output_directory = path
        self._edit.setText(path)

    def _image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        self._image_widget.set_image(cv_image)

    def shutdown_plugin(self):
        pass

    def save_settings(self, plugin_settings, instance_settings):
        #instance_settings.set_value("output_directory", self._output_directory)
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        # try:
        #     output_directory = self._set_output_directory(instance_settings.value("output_directory"))
        # except:
        #     pass
        pass
