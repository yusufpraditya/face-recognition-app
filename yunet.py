import numpy as np
import cv2

class YuNet:
    def __init__(self, model_path, input_size=[640, 480], score_threshold=0.7, nms_threshold=0.3, top_k=5000):
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
    
    def detect(self, image):
        faces = self.model.detect(image)
        if faces[1] is not None:
            for idx, face, in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                for i in range(len(coords)):
                    if coords[i] < 0:
                        coords[i] = 0
                cv2.rectangle(image, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness=2)
                return image
        
        
        
        
    