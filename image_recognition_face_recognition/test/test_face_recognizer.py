#!/usr/bin/env python

from pathlib import Path

import unittest
import cv2
import numpy as np
import torch
from parameterized import parameterized
from image_recognition_face_recognition.face_recognizer import FaceRecognizer, RecognizedFace, ROI


class TestFaceRecognizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create a FaceRecognizer object for testing
        """
        cls.face_recognizer = FaceRecognizer(device="cuda:0" if torch.cuda.is_available() else "cpu")

    def test_get_embedding(self):
        """
        Test the get_embedding method dimensions
        """
        # Create a dummy image tensor [batch_size, C, H, W]
        dummy_image = torch.randn(1, 3, 160, 160)
        embedding = self.face_recognizer._get_embedding(dummy_image)
        self.assertEqual(embedding.shape, (1, 512))

    def test_get_recognized_face(self):
        """
        Test the get_recognized_face method
        """
        # Create a dummy image and bounding box
        dummy_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        dummy_bbox = np.array([10, 10, 50, 50])
        recognized_face = self.face_recognizer._get_recognized_face(dummy_image, dummy_bbox)
        self.assertIsInstance(recognized_face, RecognizedFace)
        self.assertEqual(recognized_face.roi.x_offset, 10)
        self.assertEqual(recognized_face.roi.y_offset, 10)
        self.assertEqual(recognized_face.roi.width, 40)
        self.assertEqual(recognized_face.roi.height, 40)

    def test_update_with_categorical_distribution(self):
        """
        Test the update_with_categorical_distribution method with different thresholds
        """
        # Create a dummy recognized face
        dummy_image = torch.randn(3, 160, 160)
        dummy_roi = ROI(10, 10, 50, 50)
        recognized_face = RecognizedFace(dummy_image, dummy_roi)
        updated_face = self.face_recognizer._update_with_categorical_distribution(recognized_face)
        self.assertIsInstance(updated_face, RecognizedFace)

        # Test with different distances and thresholds
        self.face_recognizer._distance_threshold = 0.5
        updated_face = self.face_recognizer._update_with_categorical_distribution(recognized_face)
        self.assertIsInstance(updated_face, RecognizedFace)

    @parameterized.expand([
        ("doc/1.jpg", 1),
        ("doc/example.png", 7),
    ])
    def test_detect(self, image_name, expected_faces):
        """
        Test the detect method with different images (single/multiple faces)
        """
        image_path = Path(__file__).parent.parent / image_name
        image = cv2.imread(str(image_path))
        self.assertIsNotNone(image, f"Image not found at {image_path}")
        recognized_faces = self.face_recognizer.detect(image)
        self.assertIsInstance(recognized_faces, list)
        self.assertEqual(len(recognized_faces), expected_faces)

    def test_detect_no_faces(self):
        # Test with an image containing no faces
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        recognized_faces = self.face_recognizer.detect(image)
        self.assertEqual(len(recognized_faces), 0)

    def test_get_min_l2_distance(self):
        """
        Test the get_min_l2_distance method with different embeddings
        """
        # Create dummy embeddings
        old_embeddings = [np.random.randn(512) for _ in range(5)]
        new_embedding = np.random.randn(512)
        min_distance = self.face_recognizer._get_min_l2_distance(old_embeddings, new_embedding)
        self.assertIsInstance(min_distance, float)

        # Test with embeddings that are very close to each other
        old_embeddings = [np.ones(512) for _ in range(5)]
        new_embedding = np.ones(512) + 1e-6
        min_distance = self.face_recognizer._get_min_l2_distance(old_embeddings, new_embedding)
        self.assertAlmostEqual(min_distance, 0.0, places=4)

        # Test with embeddings that are very far apart
        old_embeddings = [np.ones(512) for _ in range(5)]
        new_embedding = np.zeros(512)
        min_distance = self.face_recognizer._get_min_l2_distance(old_embeddings, new_embedding)
        self.assertGreater(min_distance, 0.0)

    def test_generate_label(self):
        """
        Test the label generation and uniqueness
        """
        label1 = self.face_recognizer._generate_label()
        label2 = self.face_recognizer._generate_label()
        self.assertIsInstance(label1, str)
        self.assertIsInstance(label2, str)
        self.assertNotEqual(label1, label2)


if __name__ == "__main__":
    unittest.main()
