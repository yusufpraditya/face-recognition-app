import numpy as np
import cv2
import math
from PIL import Image

class YuNet:
    def __init__(self, model_path, input_size=[320, 320], score_threshold=0.7, nms_threshold=0.3, top_k=1):
        self.tm = cv2.TickMeter()
        self.model_path = model_path
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        
        self.model = cv2.FaceDetectorYN.create(
            model = self.model_path,
            config="",
            input_size=self.input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )

    def set_threshold(self, score_threshold):
        self.score_threshold = score_threshold
        self.model = cv2.FaceDetectorYN.create(
            model = self.model_path,
            config="",
            input_size=self.input_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
            top_k=self.top_k
        )
    
    def set_input_size(self, input_size):
        self.model.setInputSize(tuple(input_size))
    
    def crop_face(self, image):
        cropped_face = image.copy()
        faces = self.model.detect(image)
        if faces[1] is not None:
            for det in faces[1]:
                x, y, w, h = np.maximum(det[0:4].astype(np.int32), 0)
                cropped_face = cropped_face[y:y + h, x:x + w]               
                return cropped_face
        else:
            return None
        
    def visualize(self, image):
        detected_face = image.copy()
        faces = self.model.detect(image)
        if faces[1] is not None:
            for det in faces[1]:
                x, y, w, h = np.maximum(det[0:4].astype(np.int32), 0)
                start_point = (x, y)
                end_point = (x + w, y + h)
                rectangle_color = (0, 255, 0)
                cv2.rectangle(detected_face, start_point, end_point, rectangle_color, thickness=2)
                return detected_face
        else:
            return None

    def align_face(self, image, landmarks):
        if landmarks is not None and len(image) != 0:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            if left_eye[1] < right_eye[1]:
                third_point = (right_eye[0], left_eye[1])
                direction = -1
            else:
                third_point = (left_eye[0], right_eye[1])
                direction = 1

            a = self.euclidean_distance(left_eye, third_point)
            b = self.euclidean_distance(right_eye, left_eye)
            c = self.euclidean_distance(right_eye, third_point)
            cos_a = (b*b + c*c - a*a) / (2*b*c)
            angle = (np.arccos(cos_a) * 180) / math.pi
            if direction == -1:
                angle = 90 - angle
            direction = -1 * direction
            # ERROR!!!!!!!
            new_img = Image.fromarray(image)
            return np.array(new_img.rotate(direction * angle))
        else:
            return None

    def euclidean_distance(self, a, b):
        x1 = a[0]; y1 = a[1]
        x2 = b[0]; y2 = b[1]
        return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

        
    
    
        
        
        
        
    