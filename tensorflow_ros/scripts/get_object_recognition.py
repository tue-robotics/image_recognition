#!/usr/bin/env python
import argparse
from tensorflow_ros.object_recognizer import ObjectRecognizer
import cv2

# Assign description to the help doc
parser = argparse.ArgumentParser(description='Get image classifications via some TensorFlow network')

# Add arguments
parser.add_argument('-i', '--image', type=str, help='Image',
                    required=True)
parser.add_argument('-g', '--graph_path', type=str, help='Path to a trained TensorFlow output_graph.pb network',
                    required=False,
                    default="output_graph.pb")
parser.add_argument('-l', '--labels_path', type=str, help='Path to a file with the labels into which the network classifies, eg. output_labels.txt',
                    required=False,
                    default="output_labels.pb")
# parser.add_argument('-v', '--verbose', help="Increase output verbosity", action="store_true")
args = parser.parse_args()

# Read the image
img = cv2.imread(args.image)

object_recognizer = ObjectRecognizer(args.graph_path, args.labels_path)

# Pretty print the output
try:
    print object_recognizer.classify(img)
except Exception as e:
    print "An error occurred: %s" % e