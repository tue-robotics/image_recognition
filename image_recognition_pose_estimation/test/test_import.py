#!/usr/bin/env python

import unittest


class TestImport(unittest.TestCase):
    def test_import(self):
        """
        If no exception is raised, this test will succeed
        """
        import image_recognition_pose_estimation


if __name__ == "__main__":
    unittest.main()
