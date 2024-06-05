
import numpy as np
from sklearn.cluster import KMeans

import pandas as pd

class ColorExtractor(object):
    def __init__(self, total_colors=3, dominant_range=10):
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

    # function to calculate minimum distance from all colors and get the most matching color
    @staticmethod
    def getColorName(R, G, B):
        # Reading csv file with pandas and giving names to each column
        index = ["color", "color_name", "hex", "R", "G", "B"]
        csv = pd.read_csv('colors.csv', names=index, header=None)
        minimum = 10000
        for i in range(len(csv)):
            d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
            if (d <= minimum):
                minimum = d
                cname = csv.loc[i, "color_name"]
        return cname

    def recognize(self, img):
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
        factor_counts = 100.0 / sum(counts_l)
        percentages = [factor_counts * counts_l[ix] for ix in reversed(sort_ix)]
        colors = list()
        sort_ix = sort_ix[::-1]

        for i, cluster_center in enumerate(kmeans.cluster_centers_[sort_ix]):
            colors.append(ColorExtractor.getColorName(R=cluster_center[2], G=cluster_center[1], B=cluster_center[0]))

        dominant_colors.append((colors[0], percentages[0]))

        for color, percentage in zip(colors[1:], percentages[1:]):
            if percentages[0] - percentage > self._dominant_range:
                break  # Percentage are ordered, so when out of range, rest is also out of range
            if colors[0] != color:
                dominant_colors.append((color, percentage))

        return dominant_colors
