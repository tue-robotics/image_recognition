import colorsys as cs

import numpy as np
from sklearn.cluster import KMeans


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
            hue, sat, val = cs.rgb_to_hsv(cluster_center[2] / 255.0, cluster_center[1] / 255.0,
                                          cluster_center[0] / 255.0)
            hue *= 360
            if val < 0.3:
                colors.append('black')
            elif sat < 0.4:
                if val < 0.3:
                    colors.append('black')
                elif val > 0.6:
                    colors.append('white')
                else:
                    colors.append('grey')
            elif hue < 15:
                colors.append('red')
            elif hue < 40:
                colors.append('orange')
            elif hue < 65:
                colors.append('yellow')
            elif hue < 85:
                colors.append('light green')
            elif hue < 155:
                colors.append('green')
            elif hue < 175:
                colors.append('cyan')
            elif hue < 195:
                colors.append('light blue')
            elif hue < 265:
                colors.append('blue')
            elif hue < 290:
                colors.append('purple')
            elif hue < 340:
                colors.append('pink')
            else:
                colors.append('red')

        dominant_colors.append((colors[0], percentages[0]))

        for color, percentage in zip(colors[1:], percentages[1:]):
            if percentages[0] - percentage > self._dominant_range:
                break  # Percentage are ordered, so when out of range, rest is also out of range
            if colors[0] != color:
                dominant_colors.append((color, percentage))

        return dominant_colors
