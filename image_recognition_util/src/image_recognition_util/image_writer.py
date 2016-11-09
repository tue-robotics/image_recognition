import cv2
import os
import datetime


def write_annotated(dir_path, image, label, verified=False):
    """
    Write an image with an annotation to a folder
    :param dir_path: The base directory we are going to write to
    :param image: The OpenCV image
    :param label: The label that is used for creating the sub directory if not exists
    :param verified: Whether we are sure the label is correct
    """

    if dir_path is None:
        return False

    # Check if path exists, otherwise created it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Check if path exists, otherwise created it
    annotated_dir = dir_path + "/annotated"
    if not os.path.exists(annotated_dir):
        os.makedirs(annotated_dir)

    # Check if path exists, otherwise created it
    annotated_verified_unverified_dir = annotated_dir + "/verified" if verified else annotated_dir + "/unverified"
    if not os.path.exists(annotated_verified_unverified_dir):
        os.makedirs(annotated_verified_unverified_dir)

    # Check if path exists, otherwise created it
    label_dir = annotated_verified_unverified_dir + "/" + label
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    filename = "%s/%s.jpg" % (label_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"))
    cv2.imwrite(filename, image)

    return True


def write_raw(dir_path, image):
    """
    Write an image to a file (path) with the label as subfolder
    :param dir_path: The base directory we are going to write to
    :param image: The OpenCV image
    """

    if dir_path is None:
        return False

    # Check if path exists, otherwise created it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Check if path exists, otherwise created it
    raw_dir = dir_path + "/raw"
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    filename = "%s/%s.jpg" % (raw_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"))
    cv2.imwrite(filename, image)

    return True