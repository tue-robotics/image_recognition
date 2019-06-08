import cv2
import numpy as np
import os.path

from wide_resnet import WideResNet


class AgeGenderEstimator(object):
    def __init__(self, weights_file_path, img_size=64, depth=16, width=8):
        """
        Estimate the age and gender of the incoming image

        :param weights_file_path: path to a pre-trained keras network
        """
        weights_file_path = os.path.expanduser(weights_file_path)

        if not os.path.isfile(weights_file_path):
            raise IOError("Weights file {}, no such file ..".format(weights_file_path))

        self._model = None
        self._weights_file_path = weights_file_path
        self._img_size = img_size
        self._depth = depth
        self._width = width

    def estimate(self, np_images):
        """
        Estimate the age and gender of the face on the image
        :param np_images a numpy array of BGR images of faces of which the gender and the age has to be estimated
            This is assumed to be segmented/cropped already!
        :returns List of estimated age and gender score ([female, male]) tuples
        """

        # Model should be constructed in same thread as the inference
        if self._model is None:
            self._model = WideResNet(self._img_size, depth=self._depth, k=self._width)()
            self._model.load_weights(self._weights_file_path)

        faces = np.empty((len(np_images), self._img_size, self._img_size, 3))
        for i, np_image in enumerate(np_images):
            if not isinstance(np_image, np.ndarray):
                raise ValueError("np_image is not a numpy.ndarray but {t}".format(t=type(np_image)))

            if np_image.ndim != 3:  # x, y, channel
                raise ValueError("Shape of image is {sp}. Cannot classify this, shape must be (?, ?, 3). "
                                 "First 2 dimensions can are not constrained, "
                                 "but image must have 3 channels (B, G, R)".format(sp=np_image.shape))

            faces[i, :, :, :] = cv2.resize(np_image, (self._img_size, self._img_size))

        results = self._model.predict(faces)
        if not results:
            return []

        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        return zip(predicted_ages, predicted_genders)
