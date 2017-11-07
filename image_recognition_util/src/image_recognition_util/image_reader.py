import cv2
import os
import fnmatch


def read_annotated(dir_path, patterns=["*.jpg", "*.png", "*.jpeg"]):
    """
    Read annotated images from a directory. This reader assumes that the images in this directory are separated in
    different directories with the label name as directory name. The method returns a generator of the label (string)
    and the opencv image.
    :param dir_path: The base directory we are going to write read
    :param patterns: Patterns of the images the reader should match
    """
    for label in os.listdir(dir_path):
        for root, dirs, files in os.walk(dir_path):
            for basename in files:
                for pattern in patterns:
                    if fnmatch.fnmatch(basename, pattern):
                        yield label, cv2.imread(os.path.join(root, basename))
