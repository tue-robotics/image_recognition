from .face_client import FaceClient
import cv2
from timeout import Timeout


class Attribute:
    def __init__(self, value=0, confidence=0):
        """
        Face Attribute class
        :param value: The value of the attribute
        :param confidence: Confidence level
        """
        self.value = value
        self.confidence = confidence

    def __repr__(self):
        return "[%s, %s]" % (self.value, self.confidence)


class SkyFaceProperties:
    def __init__(self):
        """
        Sky Face properties object
        """
        self.age_est = Attribute()
        self.eyes = Attribute()
        self.gender = Attribute()
        self.glasses = Attribute()
        self.lips = Attribute()
        self.mood = Attribute()

    def __repr__(self):
        return "FaceProperties(age=%s, eyes=%s, gender=%s, glasses=%s, lips=%s, mood=%s)" % (
            self.age_est, self.eyes, self.gender, self.glasses, self.lips, self.mood
        )


class Skybiometry:
    def __init__(self, api_key, api_secret):
        """
        Python wrapper for the Skybiometry API
        :param api_key: The authorization key of the Skybiometry API
        :param api_secret: The secret of the Skybiometry API
        """
        self._face_client = FaceClient(api_key, api_secret)

    def _external_request_with_timeout(self, buffers, timeout):
        """
        The actual request with a timeout
        :param buffers: Image buffers
        :param timeout: specified timeout
        :return: query result
        """
        timeout_function = Timeout(self._face_client.faces_recognize, timeout)
        return timeout_function(buffers)

    def get_face_properties(self, images, timeout):
        """
        Returns a SkyFace detections list based on a list of images
        :param images: List of input images (Faces)
        :param timeout: Request timeout
        :return: The SkyFaces with their properties
        """
        buffers = [cv2.imencode('.jpg', image)[1].tostring() for image in images]

        try:
            response = self._external_request_with_timeout(buffers, timeout)
        except Exception as e:
            raise Exception("Skybiometry API call failed:", e)

        if not "photos" in response:
            raise Exception("Skybiometry API call, 'photos' not found in response:", response)

        photos = response["photos"]

        if len(photos) != len(buffers):
            raise Exception("Skybiometry API call, result length != images length:", response)

        fps = []
        for photo in photos:
            attrs = photo["tags"][0]["attributes"]
            fp = SkyFaceProperties()
            for name, attr in attrs.iteritems():
                if hasattr(fp, name):
                    setattr(fp, name, Attribute(attr["value"], attr["confidence"] / 100.0))
            fps.append(fp)

        return fps
