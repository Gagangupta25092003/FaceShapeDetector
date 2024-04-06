from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np
import random

class FacialLandmarkDetection:
    def __init__(self, imagePath) -> None:
        self.imagePath = imagePath
        self.image = cv2.imread(imagePath)
        self.height, self.width, _ = self.image.shape
        
    
    def load(self, randomVerticalFlip=False):
        self.img = mp.Image.create_from_file(self.imagePath)
        self.base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        self.options = vision.FaceLandmarkerOptions(base_options=self.base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(self.options)
        self.detector_result = self.detector.detect(self.img)
        self.faceLandmarks = self.detector_result.face_landmarks[0]
        self.points = []
        for point in self.faceLandmarks:
            x = point.x * self.width
            y = point.y * self.height
            self.points.append([x, y])
        centralPoint = self.points[5]
        self.points = np.array(self.points)
        self.points -= centralPoint # Centralizing All Faces have origin at Nose Point
        distance = math.sqrt((self.points[0][0] - self.points[152][0])**2 + (self.points[0][1] - self.points[152][1])**2 )
        self.points /= distance # Scalling all Faces to a comman Scale
        
        if randomVerticalFlip == True and random.randint(0, 1):
            self.points[:, 1] *= -1 
        return self.points
    