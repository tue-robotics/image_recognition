from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 
import cv2


def _convert_cv_to_qt_image(cv_image):
    """
    Method to convert an opencv image to a QT image
    :param cv_image: The opencv image
    :return: The QT Image
    """
    cv_image = cv_image.copy() # Create a copy
    height, width, byte_value = cv_image.shape
    byte_value = byte_value * width
    cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB, cv_image)

    return QImage(cv_image, width, height, byte_value, QImage.Format_RGB888)


def _get_roi_from_rect(rect):
    """
    Returns the ROI from a rectangle, the rectangle can have the top and bottom flipped
    :param rect: Rect to get roi from
    :return: x, y, width, height of ROI
    """
    x_min = min(rect.topLeft().x(), rect.bottomRight().x())
    y_min = min(rect.topLeft().y(), rect.bottomRight().y())
    x_max = max(rect.topLeft().x(), rect.bottomRight().x())
    y_max = max(rect.topLeft().y(), rect.bottomRight().y())

    return x_min, y_min, x_max - x_min, y_max - y_min


class ImageWidget(QWidget):

    def __init__(self, parent, image_roi_callback, clear_on_click=False):
        """
        Image widget that allows drawing rectangles and firing a image_roi_callback
        :param parent: The parent QT Widget
        :param image_roi_callback: The callback function when a ROI is drawn
        """
        super(ImageWidget, self).__init__(parent)
        self._cv_image = None
        self._qt_image = QImage()

        self.clip_rect = QRect(0, 0, 0, 0)
        self.dragging = False
        self.drag_offset = QPoint()
        self.image_roi_callback = image_roi_callback

        self.detections = []
        self._clear_on_click = clear_on_click

    def paintEvent(self, event):
        """
        Called every tick, paint event of QT
        :param event: Paint event of QT
        """
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self._qt_image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.cyan, 5.0))
        painter.drawRect(self.clip_rect)

        painter.setFont(QFont('Decorative', 10))
        for rect, label in self.detections:
            painter.setPen(QPen(Qt.magenta, 5.0))
            painter.drawRect(rect)

            painter.setPen(QPen(Qt.magenta, 5.0))
            painter.drawText(rect, Qt.AlignCenter, label)

        painter.end()

    def get_roi_image(self):
        # Flip if we have dragged the other way
        x, y, width, height = _get_roi_from_rect(self.clip_rect)

        return self._cv_image[y:y + height, x:x + width]

    def set_image(self, image):
        """
        Sets an opencv image to the widget
        :param image: The opencv image
        """
        self._cv_image = image
        self._qt_image = _convert_cv_to_qt_image(image)
        self.update()

    def add_detection(self, x, y, width, height, label):
        """
        Adds a detection to the image
        :param x: ROI_X
        :param y: ROI_Y
        :param width: ROI_WIDTH
        :param height: ROI_HEIGHT
        :param label: Text to draw
        """
        roi_x, roi_y, roi_width, roi_height = _get_roi_from_rect(self.clip_rect)
        self.detections.append((QRect(x+roi_x, y+roi_y, width, height), label))

    def clear(self):
        self.detections = []
        self.clip_rect = QRect(0, 0, 0, 0)

    def get_roi(self):
        return _get_roi_from_rect(self.clip_rect)

    def mousePressEvent(self, event):
        """
        Mouspress callback
        :param event: mouse event
        """
        # Check if we clicked on the img
        if event.pos().x() < self._qt_image.width() and event.pos().y() < self._qt_image.height():
            if self._clear_on_click:
                self.clear()
            self.clip_rect.setTopLeft(event.pos())
            self.clip_rect.setBottomRight(event.pos())
            self.dragging = True

    def mouseMoveEvent(self, event):
        """
        Mousemove event
        :param event: mouse event
        """
        if not self.dragging:
            return

        self.clip_rect.setBottomRight(event.pos())
        
        self.update()
    
    def mouseReleaseEvent(self, event):
        """
        Mouse release event
        :param event: mouse event
        """
        if not self.dragging:
            return

        roi_image = self.get_roi_image()
        if roi_image is not None:
            self.image_roi_callback(roi_image)

        self.dragging = False
