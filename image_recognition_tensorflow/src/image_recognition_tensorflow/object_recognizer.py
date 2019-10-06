"""Wrapper class around a TensorFlow neural net to classify an image"""

# Tensorflow
import tensorflow as tf
import numpy as np

# Threading in TensorFlow is interesting...
# First: ```finalize()``` the graph
# Then: Create the ```self.session``` in the main thread
# In the function called in the background: use ```with self.session.as_default() as sess:```
# See https://stackoverflow.com/questions/45093688/how-to-understand-sess-as-default-and-sess-graph-as-default


class ObjectRecognizer(object):
    def __init__(self, graph_path, labels_path, input_tensor, output_tensor):
        """
        Recognize objects via a given TensorFlow pre-trained neural net.
        The configuration oof the in/output tensor names are preset to those of a (retrained) Inception_v3 network

        :param graph_path: path to a pre-trained TensorFlow graph.pb file, e.g. output_graph.pb
        :param labels_path: path to a file containing labels. One label per line, order should match that of the graph
        :param input_tensor: Which tensor to feed the numpy-formatted image in?
        :param output_tensor: Which tensor to extract the result from?
        """
        self.labels = self._read_labels(labels_path)
        self.input_tensor_name = input_tensor
        self.output_tensor_name = output_tensor

        with open(graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            tf.compat.v1.get_default_graph().finalize()  # Make the graph read-only, safe to use from any thread

        self.session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())

        # This is only done to 'statically' check that the given tensor actually exists.
        # Not really statically but directly at startup, not when you need a first recognition
        # The Checking the tensor shapes is really a runtime thing for now
        self._input_tensor = self.session.graph.get_tensor_by_name(self.input_tensor_name)
        self._output_tensor = self.session.graph.get_tensor_by_name(self.output_tensor_name)

    def classify(self, np_image):
        """
        Classify an image into one of the labels at the label_path
        :param np_image a numpy array representing the image to be classified.
            This is assumed to be segmented/cropped already!
        :returns prediction scores for each label (ordered)
        """

        if not isinstance(np_image, np.ndarray):
            raise ValueError("np_image is not a numpy.ndarray but {t}".format(t=type(np_image)))

        if np_image.ndim != 3:  # x, y, channel
            raise ValueError("Shape of image is {sp}. Cannot classify this, shape must be (?, ?, 3). "
                             "First 2 dimensions can are not constrained, "
                             "but image must have 3 channels (B,G,R)".format(sp=np_image.shape))
        
        # Actually we want the image in RGB therefore we transform it
        np_image = np_image[:,:,::-1]
        
        # Open tf session
        with self.session.as_default() as sess:  # Be able to call this function from any thread
            # Open Image and perform prediction
            try:
                predictions = sess.run(self._output_tensor,
                                       feed_dict={self.input_tensor_name: np_image})
                predictions = np.squeeze(predictions)
            except Exception as e:
                raise Exception("Failed to run tensorflow session: %s", e)

            return list(predictions)

    @staticmethod
    def _read_labels(labels_path):
        with open(labels_path, 'rb') as f:
            labels = [label for label in f.read().split("\n") if label]  # Skip empty lines
        if not labels:
            raise ValueError("Empty labels, will not be able to map predictions to labels")
        return labels
