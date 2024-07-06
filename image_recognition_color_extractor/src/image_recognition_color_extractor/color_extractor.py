from pathlib import Path
from typing import Optional

import numpy as np
import rospy
from sklearn.cluster import KMeans

import pandas as pd


class ColorExtractor:
    def __init__(self, colors_config_file: Path, total_colors: int = 3, dominant_range: int = 10):
        """
        Constructor

        :param total_colors: Number of colors an image is split into
        :type total_colors: int
        :param dominant_range: range of percentage points relative to the most dominant color. All other colors in this
        range are also returned.
        :type dominant_range: int
        """
        self._total_colors = total_colors
        self._dominant_range = dominant_range
        # Reading csv file with pandas and giving names to each column
        column_names = ["color", "color_name", "hex", "R", "G", "B"]
        if not colors_config_file.exists():
            raise ValueError(f"Colors config file '{colors_config_file}' doesn't exist")
        self._colors_csv = pd.read_csv(colors_config_file, names=column_names, header=None)

    # function to calculate minimum distance from all colors and get the most matching color
    def get_color_name(self, red: int, green: int, blue: int) -> Optional[str]:
        """
        Get the name of closest color

        :param red: red value [0-255]
        :param green: green value [0-255]
        :param blue: blue value [0-255]

        :return: Optional color name
        """
        min_distance = 1e9
        cname = None
        for i in range(len(self._colors_csv)):
            d = abs(red - int(self._colors_csv.loc[i, "R"])) + abs(green - int(self._colors_csv.loc[i, "G"])) + abs(blue - int(self._colors_csv.loc[i, "B"]))
            if d <= min_distance:
                min_distance = d
                cname = self._colors_csv.loc[i, "color_name"]

        return cname

    def recognize(self, img: np.ndarray):
        """
        Extract the most dominant color(s)

        :param img: image to analyse
        :type img: cv2.image
        :return: List of tuples of color label and percentage (0-100)
        :rtype: list[tuple]
        """
        dominant_colors = list()

        height, width, dim = img.shape
        if height * width <= self._total_colors:
            raise RuntimeError("Total image pixels < requested total dominant colors. No dominant colors detected")

        img_vec = np.reshape(img, [height * width, dim])

        kmeans = KMeans(n_clusters=self._total_colors)
        kmeans.fit(img_vec)
        unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)

        sort_ix = np.argsort(counts_l)
        sort_ix = sort_ix[::-1]  # Reverse the list
        factor_counts = 100.0 / sum(counts_l)
        percentages = [factor_counts * counts_l[ix] for ix in sort_ix]
        colors = []

        for cluster_center in kmeans.cluster_centers_[sort_ix]:
            colors.append(self.get_color_name(red=cluster_center[2], green=cluster_center[1], blue=cluster_center[0]))

        if colors:
            dominant_colors.append((colors[0], percentages[0]))

        for color, percentage in zip(colors[1:], percentages[1:]):
            if percentages[0] - percentage > self._dominant_range:
                break  # Percentage are ordered, so when out of range, rest is also out of range
            if colors[0] != color:
                dominant_colors.append((color, percentage))

        return dominant_colors
