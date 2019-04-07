import time
from contextlib import closing
from multiprocessing import Pool

import cv2
import math
import rospy
from cv_bridge import CvBridge
from image_recognition_msgs.msg import Recognition, FaceProperties
from image_recognition_msgs.srv import Recognize, GetFaceProperties
from sensor_msgs.msg import RegionOfInterest

from image_recognition_util import image_writer


def _threaded_srv(args):
    """
    Required for calling service in parallel
    """
    srv, kwarg_dict = args
    result = srv(**kwarg_dict)
    del args
    return result


def _get_and_wait_for_services(service_names, service_class, suffix=""):
    services = {s: rospy.ServiceProxy('{}{}'.format(s, suffix), service_class) for s in service_names}
    for service in services.values():
        rospy.loginfo("Waiting for service {} ...".format(service.resolved_name))
        service.wait_for_service()
    return services


class PeopleDetector(object):
    def __init__(self):
        self._recognize_services = _get_and_wait_for_services([
            'openpose',
            'openface'
        ], Recognize, '/recognize')

        self._face_properties_services = _get_and_wait_for_services([
            'keras'
        ], GetFaceProperties, '/get_face_properties')

        self._bridge = CvBridge()

        rospy.loginfo("People detector initialized")

    def _get_recognitions(self, img):
        args = zip(self._recognize_services.values(), [{
            "image": self._bridge.cv2_to_imgmsg(img, "bgr8")
        }] * len(self._recognize_services))
        with closing(Pool(len(self._recognize_services))) as p:  # Without closing we have a memory leak
            return dict(zip(self._recognize_services.keys(), p.map(_threaded_srv, args)))

    def _get_face_properties(self, images):
        args = zip(self._face_properties_services.values(), [{
            "face_image_array": [self._bridge.cv2_to_imgmsg(image, "bgr8") for image in images]
        }] * len(self._face_properties_services))
        with closing(Pool(len(self._face_properties_services))) as p:  # Without closing we have a memory leak
            result = dict(zip(self._face_properties_services.keys(), p.map(_threaded_srv, args)))

        return result['keras'].properties_array

    @staticmethod
    def _get_recognitions_with_label(label, recognitions):
        def _is_label_recognition(recognition):
            for p in recognition.categorical_distribution.probabilities:
                if p.label == label and p.probability > recognition.categorical_distribution.unknown_probability:
                    return True
            return False

        return [r for r in recognitions if _is_label_recognition(r)]

    @staticmethod
    def _get_face_rois_openpose(recognitions):
        nose_recognitions = PeopleDetector._get_recognitions_with_label("Nose", recognitions)
        left_ear_recognitions = PeopleDetector._get_recognitions_with_label("LEar", recognitions)
        right_ear_recognitions = PeopleDetector._get_recognitions_with_label("REar", recognitions)

        rois = []
        for nose_recognition in nose_recognitions:
            # We assume a vertical head here
            left_size = 50
            right_size = 50
            try:
                left_ear_recognition = next(
                    r for r in left_ear_recognitions if r.group_id == nose_recognition.group_id)
                left_size = math.hypot(left_ear_recognition.roi.x_offset - nose_recognition.roi.x_offset,
                                       left_ear_recognition.roi.y_offset - nose_recognition.roi.y_offset)
            except StopIteration:
                pass
            try:
                right_ear_recognition = next(
                    r for r in right_ear_recognitions if r.group_id == nose_recognition.group_id)
                right_size = math.hypot(right_ear_recognition.roi.x_offset - nose_recognition.roi.x_offset,
                                        right_ear_recognition.roi.y_offset - nose_recognition.roi.y_offset)
            except StopIteration:
                pass

            size = left_size + right_size
            width = int(size)
            height = int(math.sqrt(2) * size)
            rois.append(RegionOfInterest(
                x_offset=max(0, int(nose_recognition.roi.x_offset - .5 * width)),
                y_offset=max(0, int(nose_recognition.roi.y_offset - .5 * height)),
                width=width,
                height=height
            ))
        return rois

    @staticmethod
    def _get_container_recognition(roi, recognitions, padding_factor=0.1):
        x = roi.x_offset + .5 * roi.width
        y = roi.y_offset + .5 * roi.height

        def _point_in_roi(x, y, roi):
            return roi.x_offset <= x <= roi.x_offset + roi.width and roi.y_offset <= y <= roi.y_offset + roi.height

        best = None
        for r in recognitions:
            if _point_in_roi(x, y, r.roi):
                if best:
                    avg_x = r.roi.x_offset + .5 * r.roi.width
                    avg_y = r.roi.y_offset + .5 * r.roi.height
                    best_avg_x = best.roi.x_offset + .5 * best.roi.width
                    best_avg_y = best.roi.y_offset + .5 * best.roi.height
                    if math.hypot(avg_x - x, avg_y - y) > math.hypot(best_avg_x - x, best_avg_y - y):
                        continue
                best = r
        if not best:
            best = Recognition(roi=roi)

        best.roi.x_offset = int(max(0, best.roi.x_offset - padding_factor * best.roi.width))
        best.roi.y_offset = int(max(0, best.roi.y_offset - padding_factor * best.roi.height))
        best.roi.width = int(best.roi.width + best.roi.width * 2 * padding_factor)
        best.roi.height = int(best.roi.height + best.roi.height * 2 * padding_factor)

        return best


    @staticmethod
    def _get_best_label_from_categorical_distribution(c):
        name_probabilities = [p for p in c.probabilities if p.probability > c.unknown_probability]
        if not name_probabilities:
            return None
        return max(c.probabilities, key=lambda p: p.probability)

    @staticmethod
    def _image_from_roi(image, roi):
        return image[roi.y_offset:roi.y_offset + roi.height, roi.x_offset:roi.x_offset + roi.width]

    @staticmethod
    def _get_best_label(recognition):
        best_p = None
        for p in recognition.categorical_distribution.probabilities:
            if p.probability > recognition.categorical_distribution.unknown_probability:
                if best_p and p.probability < best_p.probability:
                    continue
                best_p = p
        if best_p:
            return best_p.label
        else:
            return None

    @staticmethod
    def _face_properties_to_label(face_properties):
        return "{} (age={})".format("MALE" if face_properties.gender == FaceProperties.MALE else "FEMALE",
                                    face_properties.age)

    def recognize(self, image):
        start_recognize = time.time()
        recognitions = self._get_recognitions(image)
        rospy.logdebug("Recognize took %.4f seconds", time.time() - start_recognize)

        openpose_face_rois = PeopleDetector._get_face_rois_openpose(recognitions['openpose'].recognitions)
        face_recognitions = [PeopleDetector._get_container_recognition(openpose_face_roi,
                                                                       recognitions['openface'].recognitions)
                             for openpose_face_roi in openpose_face_rois]
        face_labels = [PeopleDetector._get_best_label(r) for r in face_recognitions]
        face_images = [PeopleDetector._image_from_roi(image, r.roi) for r in face_recognitions]
        face_properties_array = self._get_face_properties(face_images)

        cv_image = image_writer.get_annotated_cv_image(image, face_recognitions, [
            face_label if face_label else PeopleDetector._face_properties_to_label(face_properties)
            for face_label, face_properties in zip(face_labels, face_properties_array)
        ])

        return face_recognitions, cv_image
