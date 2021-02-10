import cv2
from .face_client import FaceClient
from .timeout import Timeout


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
    def __init__(self, api_key, api_secret, namespace):
        """
        Python wrapper for the Skybiometry API
        :param api_key: The authorization key of the Skybiometry API
        :param api_secret: The secret of the Skybiometry API
        :parm namespace: Namespace to use
        """
        self._face_client = FaceClient(api_key, api_secret)
        response = self._face_client.account_namespaces()
        if response["status"] != "success":
            raise ValueError("Could not retrieve namespaces for given API key and secret")
        account_namespaces = [ns["name"] for ns in response["namespaces"]]
        if namespace not in account_namespaces:
            raise ValueError("Provided namespace {} not in account_namespaces: {}".format(namespace,
                                                                                          account_namespaces))
        self._namespace = namespace

    def _external_request_with_timeout(self, buffers, timeout):
        """
        The actual request with a timeout
        :param buffers: Image buffers
        :param timeout: specified timeout
        :return: query result
        """
        timeout_function = Timeout(self._face_client.faces_recognize, timeout)
        return timeout_function(uids="operator", buffers_=buffers, namespace=self._namespace)

    def get_face_properties(self, images, timeout):
        """
        Returns a SkyFace detections list based on a list of images
        :param images: List of input images (Faces)
        :param timeout: Request timeout
        :return: The SkyFaces with their properties
        """

        buffers = ["".join(chr(int(c)) for c in cv2.imencode('.jpg', image)[1]) for image in images]

        try:
            response = self._external_request_with_timeout(buffers, timeout)
        except Exception as e:
            raise Exception("Skybiometry API call failed:", e)

        if "photos" not in response:
            raise Exception("Skybiometry API call, 'photos' not found in response:", response)

        photos = response["photos"]

        if len(photos) != len(buffers):
            raise Exception("Skybiometry API call, result length != images length:", response)

        fps = []
        for photo in photos:
            attrs = photo["tags"][0]["attributes"]
            fp = SkyFaceProperties()
            for name, attr in attrs.items():
                if hasattr(fp, name):
                    setattr(fp, name, Attribute(attr["value"], attr["confidence"] / 100.0))
            fps.append(fp)

        return fps
