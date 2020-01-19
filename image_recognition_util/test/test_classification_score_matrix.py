#!/usr/bin/env python
import unittest
from image_recognition_util.classification_score_matrix import ClassificationScoreMatrix


class TestClassificationScoreMatrix(unittest.TestCase):

    def test_classification_score_matrix_write(self):
        classification_score_matrix = ClassificationScoreMatrix(["apple", "banana"])
        classification_score_matrix.add_classification("apple", [1.0, 2.0])
        classification_score_matrix.write_to_file("/tmp/result.csv")

    def test_classification_score_matrix_wrong_label(self):
        classification_score_matrix = ClassificationScoreMatrix(["apple", "banana"])
        self.assertRaises(ValueError, classification_score_matrix.add_classification, "pear", [1.0, 2.0])

    def test_classification_score_matrix_wrong_scores(self):
        classification_score_matrix = ClassificationScoreMatrix(["apple", "banana"])
        self.assertRaises(ValueError, classification_score_matrix.add_classification, "apple", [1.0, 2.0, 3.0])


if __name__ == '__main__':
    unittest.main()
