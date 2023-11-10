import argparse
import logging
import math
import os
import sys

import cv2
import numpy
import numpy as np
import torch
import ultralytics
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO, utils

# If something is not working properly
#ultralytics.checks()
#print("My numpy version is: ", np.__version__)

# Yolo class functionality
class Yolo_wrapper():
    
    def __init__(self):

        #initialise yolov8 model
        model = YOLO('yolov8n-pose.pt') 
        self.model = model 

    def img_mapping(self,results): # map keypoints to image
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            arr = numpy.array(im)
        return arr
    
    def pose_yolo(self, img): # pose yolo for openpose_node    
        results = self.model(img)
        arr = self.img_mapping(results)
        return results, arr
    
    def pose_yolo_detect_poses(self, img): # pose yolo for detect_poses image
        results = self.model(img, save=True)
        return results
    
    def stream_yolo_detect_poses(self, classNames): # pose and obj detection yolo for detect_poses stream  

        model = YOLO('yolo-Weights/yolov8n.pt') 
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        while True:
            ret, img= cap.read()
            success, img = cap.read()
            #change self.model to model to change the neural network
            results = model(img,show = True, stream=True)

            # coordinates
            print("Results -->",results)
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence level
                    confidence = math.ceil((box.conf[0]*100))/100
                    print("Confidence --->",confidence)

                    # class name
                    cls = int(box.cls[0])

                    print("Class name -->", classNames[cls])

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    
                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                    
            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
