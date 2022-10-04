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
        cropped_face = image.copy()
        faces = self.model.detect(image)
        if faces[1] is not None:
            for det in faces[1]:
                bbox = det[0:4].astype(np.int32)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)
                cropped_face = cropped_face[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                landmarks = det[4:14].astype(np.int32).reshape((5,2))
                
                return image, landmarks
        else:
            return None, None
            

    def align_face(self, image, face):        
        if face.shape[-1] == (4 + 5 * 2):
            landmarks = face[4:].reshape(5, 2)
        else:
            raise NotImplementedError()
        warp_mat = self._getSimilarityTransformMatrix(landmarks)
        aligned_image = cv2.warpAffine(image, warp_mat, self._input_size, flags=cv2.INTER_LINEAR)
        return aligned_image
    
        
        
        
        
    