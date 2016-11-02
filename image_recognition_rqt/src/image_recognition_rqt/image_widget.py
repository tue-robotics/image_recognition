from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 
import cv2


def _convert_cv_to_qt_image(cv_image):
    cv_image = cv_image.copy() # Create a copy
    height, width, byte_value = cv_image.shape
    byte_value = byte_value * width
    cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB, cv_image)

    return QImage(cv_image, width, height, byte_value, QImage.Format_RGB888)


class ImageWidget(QWidget):
    def __init__(self, parent, image_roi_callback):
        super(ImageWidget, self).__init__(parent)   
        self._cv_image = None
        self._qt_image = QImage()

        self.clip_rect = QRect(0, 0, 0, 0)
        self.dragging = False
        self.drag_offset = QPoint()
        self.image_roi_callback = image_roi_callback

        self.detections = []

    def paintEvent(self, event):
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

    def set_image(self, image):
        self._cv_image = image
        self._qt_image = _convert_cv_to_qt_image(image)
        self.update()

    def add_detection(self, x, y, width, height, label):
        self.detections.append((QRect(x+self.clip_rect.x(), y+self.clip_rect.y(), width, height), label))

    def mousePressEvent(self, event):
        # Check if we clicked on the img
        if event.pos().x() < self._qt_image.width() and event.pos().y() < self._qt_image.height():
            self.detections = []
            self.clip_rect.setTopLeft(event.pos())
            self.clip_rect.setBottomRight(event.pos())
            self.dragging = True
    
    def mouseMoveEvent(self, event):
        if not self.dragging:
            return

        self.clip_rect.setBottomRight(event.pos())
        
        self.update()
    
    def mouseReleaseEvent(self, event):
        if not self.dragging:
            return

        roi_image = self._cv_image[self.clip_rect.y(): self.clip_rect.y() + self.clip_rect.height(),
                                self.clip_rect.x(): self.clip_rect.x() + self.clip_rect.width()]
        self.image_roi_callback(roi_image, self.clip_rect.x(), self.clip_rect.y(),
                                self.clip_rect.width(), self.clip_rect.height())

        self.dragging = False
