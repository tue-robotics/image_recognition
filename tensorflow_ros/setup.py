from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['tensorflow_ros'],
    scripts=['scripts/object_recognition_node.py', 'scripts/retrain.py'],
    package_dir={'': 'src'}
)

setup(**d)