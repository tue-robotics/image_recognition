"""Wrapper class around a TensorFlow neural net to classify an image"""

# Tensorflow
import tensorflow as tf
import numpy as np

import cv2

import operator

class ObjectRecognizer(object):
    def __init__(self, graph_path, labels_path, save_images_folder=None):
        self.save_image_folder = save_images_folder

        self.labels = self._read_labels(labels_path)

        with open(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')  # This magically stays in memory as the network tensorflow should work with

    def classify(self, np_image):
        """Classify an image into one of the labels at the label_path
        :param np_image a numpy array representing the image to be classified. This is assumed to be segmented/cropped already!
        :returns a dictionary mapping class to probability of the image being that class"""

        filename = self._save_to_file(np_image)

        # Open tf session
        with tf.Session() as sess:

            # 3. Get result tensor
            # result_tensor = sess.graph.get_tensor_by_name("softmax:0")
            result_tensor = sess.graph.get_tensor_by_name("final_result:0")

            # Open Image and perform prediction
            predictions = []
            try:
                with open(filename, 'rb') as f:
                    predictions = sess.run(result_tensor,                           # Run the inference for this tensor
                                           {'DecodeJpeg/contents:0': f.read()})     # with some tensor substituted by our input data
                    predictions = np.squeeze(predictions)
            except Exception as e:
                raise Exception("Failed to run tensorflow session: %s", e)

            # Open output_labels and construct dict from result
            result = dict(zip(self.labels, predictions))

            return result

    def classify_many(self, *np_images):
        return map(self.classify, np_images)

    def _save_to_file(self, np_image):
        """
        Save a numpy image to our tempfile
        :param np_image:
        :return:
        """

        filename = "/tmp/{id}.jpg".format(id=id(np_image))
        cv2.imwrite(filename=filename, img=np_image)

        return filename

    def _read_labels(self, labels_path):
        with open(labels_path, 'rb') as f:
            labels = f.read().split("\n")
        if not labels:
            raise ValueError("Empty labels, will not be able to map predictions to labels")
        return labels

def order_dict_by_value(d):
    sorted_result = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_result