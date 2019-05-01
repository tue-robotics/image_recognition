#!/usr/bin/env python
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sklearn.cluster import KMeans
import numpy as np
import colorsys as cs
from image_recognition_msgs.srv import ExtractColour, ExtractColourResponse
from std_msgs.msg import String


class ColourExtractor(object):
    def __init__(self):

        # services
        self.srv_extract_colour = rospy.Service('~extract_color', ExtractColour, self._extract_colour_srv)

        # initialize parameters
        self._bridge = CvBridge()

    def _extract_colour_srv(self, req):
        img = self._bridge.imgmsg_to_cv2(req.image, desired_encoding="passthrough")
        height, width, dim = img.shape
        img_vec = np.reshape(img, [height * width, dim])

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(img_vec)
        unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)

        sort_ix = np.argsort(counts_l)
        factor_counts = 100.0 / sum(counts_l)
        percentages = [factor_counts * counts_l[sort_ix[2]], factor_counts * counts_l[sort_ix[1]]]
        colours = []
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
            dominant_colours = [colours[0]]

            if percentages[0] - percentages[1] < 10 and colours[0] != colours[1]:
                dominant_colours.append(colours[1])

            resp = ExtractColourResponse()
            resp.colours = map(String, dominant_colours)
            return resp


if __name__ == "__main__":
    rospy.init_node('colour_extractor')
    colour_extractor = ColourExtractor()

    rospy.spin()