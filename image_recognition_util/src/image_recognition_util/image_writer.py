import numpy as np

import cv2
import os
import datetime


def color_map(N=256, normalized=False):
    """
    Generate an RGB color map of N different colors
    :param N : int amount of colors to generate
    :param normalized: bool indicating range of each channel: float32 in [0, 1] or int in [0, 255]
    :return a numpy.array of shape (N, 3) with a row for each color and each row is [R,G,B]
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i + 1  # skip the first color (black)
        for j in range(8):
            r |= bitget(c, 0) << 7 - j
            g |= bitget(c, 1) << 7 - j
            b |= bitget(c, 2) << 7 - j
            c >>= 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def write_estimation(dir_path, image, label, annotated_original_image=None, suffix=""):
    """
    Write estimation to a directory, for the estimation, a directory of the run will be created
    """
    if dir_path is None:
        return False

    # Check if path exists, otherwise created it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Check if path exists, otherwise created it
    estimations_dir = dir_path + "/estimations"
    if not os.path.exists(estimations_dir):
        os.makedirs(estimations_dir)

    # Make a directory of the estimation with current time
    estimation_dir = "%s/%s%s" % (estimations_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"), suffix)
    os.makedirs(estimation_dir)

    filename = "%s/%s.jpg" % (estimation_dir, label)
    cv2.imwrite(filename, image)

    if annotated_original_image is not None:
        filename = "%s/annotated_original_image.jpg" % estimation_dir
        cv2.imwrite(filename, annotated_original_image)

    return True


def write_estimations(dir_path, images, labels, annotated_original_image=None, suffix=""):
    """
    Write estimations to a directory, for each estimation cycle, a directory of the run will be created
    """
    assert len(images) == len(labels)

    if dir_path is None:
        return False

    # Check if path exists, otherwise created it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Check if path exists, otherwise created it
    estimations_dir = dir_path + "/estimations"
    if not os.path.exists(estimations_dir):
        os.makedirs(estimations_dir)

    # Make a directory of the estimation with current time
    estimation_dir = "%s/%s%s" % (estimations_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"), suffix)
    os.makedirs(estimation_dir)

    for i, (image, label) in enumerate(zip(images, labels)):
        filename = "%s/%s_%d.jpg" % (estimation_dir, label, i)
        cv2.imwrite(filename, image)

    if annotated_original_image is not None:
        filename = "%s/annotated_original_image.jpg" % estimation_dir
        cv2.imwrite(filename, annotated_original_image)

    return True


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


def write_raw(dir_path, image, subfolder_name="raw"):
    """
    Write an image to a file (path) with the label as subfolder
    :param dir_path: The base directory we are going to write to
    :param image: The OpenCV image
    :param subfolder_name: A directory within the path is created with this name
    """

    if dir_path is None:
        return False

    # Check if path exists, otherwise created it
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Check if path exists, otherwise created it
    raw_dir = os.path.join(dir_path, subfolder_name)
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    filename = "%s/%s.jpg" % (raw_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"))
    cv2.imwrite(filename, image)

    return True


def get_annotated_cv_image(cv_image, recognitions, labels=[]):
    """
    Gets an annotated CV image based on recognitions, drawin using cv.rectangle
    :param cv_image: Original cv image
    :param recognitions: List of recognitions
    :param labels: List of labels per recognition
    :return: Annotated image
    """
    annotated_cv_image = cv_image.copy()

    c_map = color_map(N=len(recognitions), normalized=True)
    for i, recognition in enumerate(recognitions):
        x_min, y_min = recognition.roi.x_offset, recognition.roi.y_offset
        x_max, y_max = x_min + recognition.roi.width, y_min + recognition.roi.height

        cv2.rectangle(annotated_cv_image, (x_min, y_min), (x_max, y_max),
                      (c_map[i, 2] * 255, c_map[i, 1] * 255, c_map[i, 0] * 255), 10)

    for i, (recognition, label) in enumerate(zip(recognitions, labels)):
        font = cv2.FONT_HERSHEY_PLAIN
        x, y = recognition.roi.x_offset, recognition.roi.y_offset
        font_scale = 1.0
        font_color = (255, 255, 255)
        bg_color = (0, 0, 0)
        line_type = 1

        (text_width, text_height) = cv2.getTextSize(label, font, fontScale=font_scale, thickness=1)[0]
        box_coords = ((x, y), (x + text_width - 2, y - text_height - 2))
        cv2.rectangle(annotated_cv_image, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
        cv2.putText(annotated_cv_image, label,
                    (x, y),
                    font,
                    font_scale,
                    font_color,
                    line_type)

    return annotated_cv_image
