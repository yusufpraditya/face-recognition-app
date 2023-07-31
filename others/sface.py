import numpy as np
import cv2

class SFace:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = cv2.FaceRecognizerSF.create(
            model=self.model_path,
            config=""
        )

    def feature(self, face):
        if face is not None:
            feature = self.model.feature(face)
            return feature
        return None
    
    def match(self, feature1, feature2):
        cosine_score = self.model.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        return cosine_score

