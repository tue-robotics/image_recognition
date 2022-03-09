from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['image_recognition_age_gender'],
    package_dir={'': 'src'}
)

setup(**d)
