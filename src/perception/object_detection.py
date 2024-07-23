import cv2
import numpy as np
import rospy

class ObjectDetection:

    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromTensorflow(model_path)

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(
            image, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        return detections
    
    def draw_detections(self, image, detections):
        pass