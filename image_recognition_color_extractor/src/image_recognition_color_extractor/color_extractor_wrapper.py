import os

import numpy as np
import rospkg

from .color_extractor.image_to_color import ImageToColor


class ColorExtractor(object):
    def __init__(self):
        pkg_path = rospkg.RosPack().get_path("image_recognition_color_extractor")
        npz = np.load(os.path.join(pkg_path, "data/color_names.npz"))

        self._image_to_color = ImageToColor(npz["samples"], npz["labels"])

    def extract_colors(self, np_images):
        return [self._image_to_color.get(np_image[..., ::-1]) for np_image in np_images]
