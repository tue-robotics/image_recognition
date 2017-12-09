import cv2
import os
import fnmatch


def read_annotated(dir_path, patterns=["*.jpg", "*.png", "*.jpeg"]):
    """
    Read annotated images from a directory. This reader assumes that the images in this directory are separated in
    different directories with the label name as directory name. The method returns a generator of the label (string)
    , the opencv image and the filename.
    :param dir_path: The base directory we are going to read
    :param patterns: Patterns of the images the reader should match
    """
    for label in os.listdir(dir_path):
        for root, dirs, files in os.walk(os.path.join(dir_path, label)):
            for basename in files:
                for pattern in patterns:
                    if fnmatch.fnmatch(basename, pattern):
                        filename = os.path.join(root, basename)
                        image = cv2.imread(filename)
                        if image is None:
                            print ">> Ignore empty image {f}".format(f=filename)
                        else:
                            yield label, image, filename
