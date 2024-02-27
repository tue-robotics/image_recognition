#!/usr/bin/env python

import argparse
import logging
import sys

import cv2

from image_recognition_pose_estimation.yolo_pose_wrapper import YoloPoseWrapper


def main(args: argparse.Namespace):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    wrapper = YoloPoseWrapper(args.pose_model, args.device, args.verbose)

    if args.mode == "image":
        # Read the image
        image = cv2.imread(args.image)
        recognitions, overlayed_image = wrapper.detect_poses(image, args.conf)

        logging.info(recognitions)
        cv2.imshow("overlayed_image", overlayed_image)

        cv2.waitKey()

    elif args.mode == "cam":
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            recognitions, overlayed_image = wrapper.detect_poses(img)
            cv2.imshow("overlayed_image", overlayed_image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect poses in an image", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--pose_model", help="Pose model to use", default="yolov8n-pose.pt", type=str)
    parser.add_argument("--device", help="Device to use", default="cuda:0", type=str)
    parser.add_argument("--conf", help="Minimal confidence level for detection", default=0.25, type=float)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If True, enables verbose output during the model's operations. Defaults to False.",
    )

    mode_parser = parser.add_subparsers(help="Mode")
    image_parser = mode_parser.add_parser("image", help="Use image mode")
    image_parser.set_defaults(mode="image")
    cam_parser = mode_parser.add_parser("cam", help="Use cam mode")
    cam_parser.set_defaults(mode="cam")

    # Image specific arguments
    image_parser.add_argument("image", help="Input image")

    args = parser.parse_args()

    sys.exit(main(args))
