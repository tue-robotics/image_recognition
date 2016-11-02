from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 


def warning_dialog(title, text):
    """
    Helper function for creating a warning dialog
    :param title: Title of the dialog
    :param text: Text of the dialog
    """
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(text)
    msg.setWindowTitle(title)
    msg.exec_()


def info_dialog(title, text):
    """
    Helper function for creating a info dialog
    :param title: Title of the dialog
    :param text: Text of the dialog
    """
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(text)
    msg.setWindowTitle(title)
    msg.exec_()


def option_dialog(title, options):
    """
    Helper function for creating an option dialog
    :param title: Title of the dialog
    :param options: Array of options
    :return: The clicked option string
    """

    class OptionDialog(QDialog):
        def __init__(self, title, options):
            super(OptionDialog, self).__init__()

            self.layout = QGridLayout()
            self.option_selector = QComboBox()
            self.option_selector.addItems(list(options))
            self.layout.addWidget(QLabel("%s:" % title), 0, 0)
            self.layout.addWidget(self.option_selector, 0, 1)
            self.ok_button = QPushButton('ok')
            self.ok_button.clicked.connect(self.accept)
            self.layout.addWidget(self.ok_button, 0, 2)

            self.setLayout(self.layout)

        def keyPressEvent(self, event):
            super(OptionDialog, self).keyPressEvent(event)
            if event.key() == Qt.Key_Return:
                self.accept()

    dlg = OptionDialog(title, options)
    if dlg.exec_():
        return str(dlg.option_selector.currentText())
    return None
