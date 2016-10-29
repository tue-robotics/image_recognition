from qt_gui.plugin import Plugin

from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import *

from tensorflow_ros import retrain, utils


def dialog(title, text, icon=QMessageBox.Information):
    msg = QMessageBox()
    msg.setIcon(icon)
    msg.setText(text)
    msg.setWindowTitle(title)
    msg.exec_()


class TrainPlugin(Plugin):
    batch = retrain.defaults.batch
    steps = retrain.defaults.steps
    output_directory = "/tmp"
    images_directory = "/tmp"

    def __init__(self, context):
        super(TrainPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Train Plugin')

        self._widget = QWidget()
        context.add_widget(self._widget)

        # Layout and attach to widget
        layout = QGridLayout()
        self._widget.setLayout(layout)

        self._edit_images_path_button = QPushButton("Edit images dir")
        self._edit_images_path_button.clicked.connect(self._get_images_directory)
        layout.addWidget(self._edit_images_path_button, 1, 1)

        self._images_path_edit = QLineEdit()
        self._images_path_edit.setDisabled(True)
        layout.addWidget(self._images_path_edit, 1, 2)

        self._edit_output_path_button = QPushButton("Edit output dir")
        self._edit_output_path_button.clicked.connect(self._get_output_directory)
        layout.addWidget(self._edit_output_path_button, 2, 1)

        self._output_path_edit = QLineEdit()
        self._output_path_edit.setDisabled(True)
        layout.addWidget(self._output_path_edit, 2, 2)

        self._train_button = QPushButton("Train that!")
        self._train_button.clicked.connect(self._train)
        layout.addWidget(self._train_button, 3, 2)

    def _set_images_directory(self, path):
        if not path:
            path = "/tmp"

        self.images_directory = path
        self._images_path_edit.setText("Using images from %s" % path)

    def _get_images_directory(self):
        self._set_images_directory(QFileDialog.getExistingDirectory(self._widget, "Select images directory"))

    def _set_output_directory(self, path):
        if not path:
            path = "/tmp"

        self.output_directory = path
        self._output_path_edit.setText("Saving train output to %s" % path)

    def _get_output_directory(self):
        self._set_output_directory(QFileDialog.getExistingDirectory(self._widget, "Select output directory"))

    def _train(self):
        model_dir = "/tmp/inception"
        utils.maybe_download_and_extract("http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz",
                                         "/tmp/inception")
        try:
            retrain.main(self.images_directory, model_dir, self.output_directory,
                         steps=self.steps, batch=self.batch)
            dialog("Retrain succes", "Succesfully retrained the top layers")
        except Exception as e:
            dialog("Retrain failed", "Something went wrong during retraining, '%s'" % str(e), QMessageBox.Warning)

    def trigger_configuration(self):
        batch, ok = QInputDialog.getInt(self._widget, "Set batch size", "Batch size", self.batch)

        # Update batch if ok
        self.batch = batch if ok else self.batch

        steps, ok = QInputDialog.getInt(self._widget, "Set steps size", "Step size", self.steps    )

        # Update batch if ok
        self.steps = steps if ok else self.steps

        self._update_configuration_title()

    def shutdown_plugin(self):
        pass

    def save_settings(self, plugin_settings, instance_settings):
        instance_settings.set_value("output_directory", self.output_directory)
        instance_settings.set_value("images_directory", self.images_directory)
        instance_settings.set_value("steps", self.steps)
        instance_settings.set_value("batch", self.batch)

    def _update_configuration_title(self):
        self._train_button.setText("Train (steps=%d, batch=%d)" % (self.steps, self.batch))

    def restore_settings(self, plugin_settings, instance_settings):
        try:
            self._set_output_directory(instance_settings.value("output_directory"))
            self._set_images_directory(instance_settings.value("images_directory"))
            self.batch = int(instance_settings.value("batch"))
            self.steps = int(instance_settings.value("steps"))
        except:
            pass
        self._update_configuration_title()
