#!/usr/bin/env python
import argparse
from tensorflow_ros.object_recognizer import ObjectRecognizer, order_dict_by_value
import cv2

import os
import pprint

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
parser.add_argument('-n', '--top_n', type=int, help='Top N results to display',
                    required=False,
                    default=1)
# parser.add_argument('-v', '--verbose', help="Increase output verbosity", action="store_true")
args = parser.parse_args()


object_recognizer = ObjectRecognizer(args.graph_path, args.labels_path)

if os.path.isfile(args.image):
    # Pretty print the output
    try:
        # Read the image
        img = cv2.imread(args.image)
        raw_result = object_recognizer.classify(img)
        ordered_result = order_dict_by_value(raw_result)

        pprint.pprint(ordered_result[:args.top_n])

    except Exception as e:
        print "An error occurred: %s" % e
elif os.path.isdir(args.image):
    dirname = args.image

    filenames = os.listdir(dirname)
    image_paths = [os.path.join(dirname, f) for f in filenames]

    images = map(cv2.imread, image_paths)

    predictions_per_image = object_recognizer.classify_many(*images)
    sorted_predictions_per_image = map(order_dict_by_value, predictions_per_image)

    # Compose a dict mapping each filename to the predictions for each class
    results = dict(zip(image_paths, [prediction[:args.top_n] for prediction in sorted_predictions_per_image]))

    pprint.pprint(results)