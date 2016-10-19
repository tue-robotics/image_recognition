from qt_gui.plugin import Plugin

from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 

class TrainPlugin(Plugin):

    def __init__(self, context):
        super(TrainPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Train Plugin')

        self._widget = QWidget()
        context.add_widget(self._widget)

    def trigger_configuration(self):
        pass

    def shutdown_plugin(self):
        pass

    def save_settings(self, plugin_settings, instance_settings):
        pass

    def restore_settings(self, plugin_settings, instance_settings):
        pass