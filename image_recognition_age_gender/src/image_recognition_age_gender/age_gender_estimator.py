import cv2
import numpy as np
import os.path

import onnxruntime

GENDER_DICT = {0: 'male', 1: 'female'}


class AgeGenderEstimator(object):
    def __init__(self, weights_file_path, img_size=64, use_gpu=False):
        """
        Estimate the age and gender of the incoming image

        :param weights_file_path: path to a pre-trained network in onnx format
        :param img_size: Images are resized to a square image of (img_size X img_size)
        :param use_gpu: Use GPU or CPU
        """
        weights_file_path = os.path.expanduser(weights_file_path)

        if not os.path.isfile(weights_file_path):
            raise IOError("Weights file {}, no such file ..".format(weights_file_path))

        self._model = None
        self._weights_file_path = weights_file_path
        self._img_size = img_size
        self._use_gpu = use_gpu

    def estimate(self, np_images):
        """
        Estimate the age and gender of the face on the image

        :param np_images a numpy array of BGR images of faces of which the gender and the age has to be estimated
            This is assumed to be segmented/cropped already!
        :returns List of estimated age and gender score ([female, male]) tuples
        """

        # Model should be constructed in same thread as the inference
        if self._model is None:
            providers = ['CPUExecutionProvider']
            if self._use_gpu:
                providers.append(
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    })),

            self._model = onnxruntime.InferenceSession(self._weights_file_path, providers=providers)

        results = []
        for np_image in np_images:
            inputs = np.transpose(cv2.resize(np_image, (self._img_size, self._img_size)), (2, 0, 1))
            inputs = np.expand_dims(inputs, 0).astype(np.float32) / 255.
            predictions = self._model.run(['output'], input_feed={'input': inputs})[0][0]
            #           age              p(male)          p(female)
            results += [(predictions[2], (predictions[0], predictions[1]))]

        return results
