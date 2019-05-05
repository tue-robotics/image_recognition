#! /usr/bin/env python

# Python libraries
import numpy as np
import colorsys as cs
from sklearn.cluster import KMeans

class ColourExtractor(object):
    def __init__(self, total_colours):
        self._total_colours = total_colours

    def recognize(self, img):
        dominant_colours = list()
        return_message = None

        height, width, dim = img.shape
        if height * width <= self._total_colours:
            return_message = "Total image pixels lesser than requested total dominant colours. No dominant colours detected"
            return dominant_colours, return_message


        img_vec = np.reshape(img, [height * width, dim])

        kmeans = KMeans(n_clusters=self._total_colours)
        kmeans.fit(img_vec)
        unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)

        sort_ix = np.argsort(counts_l)
        factor_counts = 100.0 / sum(counts_l)
        percentages = [factor_counts * counts_l[sort_ix[2]], factor_counts * counts_l[sort_ix[1]]]
        colours = list()
        sort_ix = sort_ix[::-1]

        for i, cluster_center in enumerate(kmeans.cluster_centers_[sort_ix]):
            hue, sat, val = cs.rgb_to_hsv(cluster_center[2] / 255.0, cluster_center[1] / 255.0,
                                          cluster_center[0] / 255.0)
            hue *= 360
            if val < 0.3:
                colours.append('black')
            elif sat < 0.4:
                if val < 0.3:
                    colours.append('black')
                elif val > 0.6:
                    colours.append('white')
                else:
                    colours.append('grey')
            elif hue < 15:
                colours.append('red')
            elif hue < 40:
                colours.append('orange')
            elif hue < 65:
                colours.append('yellow')
            elif hue < 85:
                colours.append('light green')
            elif hue < 155:
                colours.append('green')
            elif hue < 175:
                colours.append('cyan')
            elif hue < 195:
                colours.append('light blue')
            elif hue < 265:
                colours.append('blue')
            elif hue < 290:
                colours.append('purple')
            elif hue < 340:
                colours.append('pink')
            else:
                colours.append('red')

        # print colours
        dominant_colours.append(colours[0])

        if percentages[0] - percentages[1] < 10 and colours[0] != colours[1]:
            dominant_colours.append(colours[1])

        return dominant_colours, return_message

