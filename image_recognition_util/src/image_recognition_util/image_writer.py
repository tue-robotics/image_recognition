import datetime
import errno
import os
from typing import List, Optional

import cv2
import numpy as np
from image_recognition_msgs.msg import Recognition


def mkdir_p(directory: str) -> None:
    """
    os.makedirs() without raising an exception in case of existence

    :param directory: directory to create
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def create_estimation_dir(parent_dir: str, suffix: str) -> Optional[str]:
    """
    Create an estimation dir in parent directory

    :param parent_dir: parent directory
    :param suffix:
    :return: Created estimations directory
    """
    if parent_dir is None:
        return None

    estimation_dir = os.path.join(parent_dir, "estimations")

    # Make a directory of the estimation with current time
    estimation_dir = os.path.join(estimation_dir, "".join([datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"),
                                                           suffix]))
    mkdir_p(estimation_dir)

    return estimation_dir


def color_map(n: int = 256, normalized: bool = False) -> np.ndarray:
    """
    Generate an RGB color map of N different colors
    :param n: amount of colors to generate
    :type n: int
    :param normalized: indicating range of each channel: float32 in [0, 1] or int in [0, 255]
    :type normalized: bool
    :return a numpy.array of shape (N, 3) with a row for each color and each row is [R,G,B]
    """
    def bitget(byteval: int, idx: int) -> bool:
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((n, 3), dtype=dtype)
    for i in range(n):
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


def write_estimation(
    dir_path: str,
    image: np.ndarray,
    label: str,
    annotated_original_image: Optional[np.ndarray] = None,
    suffix: str = "",
) -> bool:
    """
    Write estimation to a directory.
    For the estimation, a directory of the run will be created

    :param dir_path: Parent directory
    :param image: Image to write
    :param label: Label of the image
    :param annotated_original_image: The entire image with annotations
    :param suffix: Suffix of the run specific directory
    :return: Success
    """

    estimation_dir = create_estimation_dir(dir_path, suffix)
    if estimation_dir is None:
        return False

    filename = os.path.join(estimation_dir, f"{label}.jpg")
    cv2.imwrite(filename, image)

    if annotated_original_image is not None:
        filename = os.path.join(estimation_dir, "annotated_original_image.jpg")
        cv2.imwrite(filename, annotated_original_image)

    return True


def write_estimations(
    dir_path: str,
    images: List[np.ndarray],
    labels: List[str],
    annotated_original_image: Optional[np.ndarray] = None,
    suffix: str = "",
    ) -> bool:
    """
    Write estimations to a directory.
    For each estimation cycle, a directory of the run will be created

    :param dir_path: Parent directory
    :param images: Image to write
    :param labels: Label of the image
    :param annotated_original_image: The entire image with annotations
    :param suffix: Suffix of the run specific directory
    :return: Success
    """
    assert len(images) == len(labels)

    estimation_dir = create_estimation_dir(dir_path, suffix)
    if not estimation_dir:
        return False

    for i, (image, label) in enumerate(zip(images, labels)):
        filename = os.path.join(estimation_dir, f"{label}_{i}.jpg")
        cv2.imwrite(filename, image)

    if annotated_original_image is not None:
        filename = os.path.join(estimation_dir, "annotated_original_image.jpg")
        cv2.imwrite(filename, annotated_original_image)

    return True


def write_annotated(dir_path: str, image: np.ndarray, label: str, verified: bool = False) -> bool:
    """
    Write an image with an annotation to a folder

    :param dir_path: The base directory we are going to write to
    :param image: The OpenCV image
    :param label: The label that is used for creating the subdirectory if not exists
    :param verified: Whether we are sure the label is correct
    :return: Success
    """

    if dir_path is None:
        return False

    annotated_dir = os.path.join(dir_path, "annotated")
    annotated_verified_unverified_dir = os.path.join(annotated_dir, "verified" if verified else "unverified")
    label_dir = os.path.join(annotated_verified_unverified_dir, label)
    mkdir_p(label_dir)

    filename = os.path.join(label_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_%f')}.jpg")
    cv2.imwrite(filename, image)

    return True


def write_raw(dir_path: str, image: np.ndarray, subfolder_name: str = "raw") -> bool:
    """
    Write an image to a file (path) with the label as subfolder
    :param dir_path: The base directory we are going to write to
    :param image: The OpenCV image
    :param subfolder_name: A directory within the path is created with this name
    :return: Success
    """

    if dir_path is None:
        return False

    raw_dir = os.path.join(dir_path, subfolder_name)
    mkdir_p(raw_dir)

    filename = os.path.join(raw_dir, f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S_%f')}.jpg")
    cv2.imwrite(filename, image)

    return True


def get_annotated_cv_image(
    cv_image: np.ndarray, recognitions: List[Recognition], labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    Gets an annotated CV image based on recognitions, drawing using cv.rectangle
    :param cv_image: Original cv image
    :param recognitions: List of recognitions
    :param labels: List of labels per recognition
    :return: Annotated image
    """
    if labels is None:
        labels = []
    annotated_cv_image = cv_image.copy()

    c_map = color_map(n=len(recognitions), normalized=True)
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
