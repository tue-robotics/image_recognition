from face_client import FaceClient
import cv2
from timeout import Timeout


class Attribute:
    def __init__(self, value=0, confidence=0):
        self.value = value
        self.confidence = confidence

    def __repr__(self):
        return "[%s, %s]" % (self.value, self.confidence)


class SkyFaceProperties:
    def __init__(self):
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
        self._face_client = FaceClient(api_key, api_secret)

    def _external_request_with_timeout(self, buffers, timeout):
        timeout_function = Timeout(self._face_client.faces_recognize, timeout)
        return timeout_function(buffers)

    def get_face_properties(self, images, timeout):
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
