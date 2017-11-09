"""Wrapper class around a TensorFlow neural net to classify an image"""

# Tensorflow
import tensorflow as tf
import numpy as np

import cv2


from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

# Threading in TensorFlow is interesting...
# First: ```finalize()``` the graph
# Then: Create the ```self.session``` in the main thread
# In the function called in the background: use ```with self.session.as_default() as sess:```
# See https://stackoverflow.com/questions/45093688/how-to-understand-sess-as-default-and-sess-graph-as-default

class ObjectRecognizer(object):
    def __init__(self, graph_path, labels_path):

        self.labels = self._read_labels(labels_path)

        with open(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            tf.get_default_graph().finalize()  # Make the graph read-only, safe to use from any thread

        self.session = tf.Session(graph=tf.get_default_graph())

    def classify(self, np_image):
        """
        Classify an image into one of the labels at the label_path
        :param np_image a numpy array representing the image to be classified.
            This is assumed to be segmented/cropped already!
        :returns a dictionary mapping class to probability of the image being that class
        """

        # Open tf session
        with self.session.as_default() as sess:  # Be able to call this function from any thread

            # Get result tensor that will eventually hold the predictions
            result_tensor = sess.graph.get_tensor_by_name("final_result:0")

            # Open Image and perform prediction
            try:
                predictions = sess.run(result_tensor,
                                       feed_dict={"Cast:0": np_image})
                predictions = np.squeeze(predictions)
            except Exception as e:
                raise Exception("Failed to run tensorflow session: %s", e)

            # Open output_labels and construct dict from result
            result = sorted(zip(self.labels, predictions), key=lambda pair: pair[1], reverse=True)

            return result

    @staticmethod
    def _read_labels(labels_path):
        with open(labels_path, 'rb') as f:
            labels = [label for label in f.read().split("\n") if label]  # Skip empty lines
        if not labels:
            raise ValueError("Empty labels, will not be able to map predictions to labels")
        return labels
