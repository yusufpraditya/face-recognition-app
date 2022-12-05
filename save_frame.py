import argparse
from time import time
import numpy as np
import cv2 as cv
from PIL import Image
import os
import datetime

backends = (cv.dnn.DNN_BACKEND_DEFAULT,
                cv.dnn.DNN_BACKEND_HALIDE,
                cv.dnn.DNN_BACKEND_INFERENCE_ENGINE,
                cv.dnn.DNN_BACKEND_OPENCV)
targets = (cv.dnn.DNN_TARGET_CPU,
            cv.dnn.DNN_TARGET_OPENCL,
            cv.dnn.DNN_TARGET_OPENCL_FP16,
            cv.dnn.DNN_TARGET_MYRIAD)

parser = argparse.ArgumentParser(description='A demo for running libfacedetection using OpenCV\'s DNN module.')
parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                    help='Choose one of computation backends: '
                            '%d: automatically (by default), '
                            '%d: Halide language (http://halide-lang.org/), '
                            '%d: Intel\'s Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), '
                            '%d: OpenCV implementation' % backends)
parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                    help='Choose one of target computation devices: '
                            '%d: CPU target (by default), '
                            '%d: OpenCL, '
                            '%d: OpenCL fp16 (half-float precision), '
                            '%d: VPU' % targets)
parser.add_argument('--input_mean',default=127.5, type=float,help='input_mean')
parser.add_argument('--input_std',default=127.5, type=float,help='input standard deviation')
# Location
parser.add_argument('--input', '-i', help='Path to the image. Omit to call default camera')
parser.add_argument('--model', '-m', default="face_detection_yunet.onnx", type=str, help='Path to .onnx model file.')
parser.add_argument('--face_recognition_model', default="face_recognition_sface.onnx", type=str, help='Path to .onnx model file.')
# Inference parameters
parser.add_argument('--score_threshold', default=0.9, type=float, help='Threshold for filtering out faces with conf < conf_thresh.')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='Threshold for non-max suppression.')
parser.add_argument('--top_k', default=5000, type=int, help='Keep keep_top_k for results outputing.')
args = parser.parse_args()

now = datetime.datetime.now()
folder_name = now.strftime(args.input.split('.')[0] + "_%H%M%S/")
target = os.getcwd() + '/saved_frame/' + folder_name

os.makedirs(target)
  

if __name__ == '__main__':

    yunet = cv.FaceDetectorYN.create(
        model=args.model,
        config='',
        input_size=(320, 320),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        top_k=5000,
        backend_id=args.backend,
        target_id=args.target
    )
    recognizer = cv.FaceRecognizerSF.create(args.face_recognition_model,"")  
    #frame_w = 640
    #frame_h = 360
    
    if args.input == "0":
        cap = cv.VideoCapture(int(args.input))
    else:
        cap = cv.VideoCapture(args.input)
    scale_percent = 100 # percent of original size
    frame_w = int(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) * scale_percent / 100)
    frame_h = int(int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) * scale_percent / 100)
    dim = (frame_w, frame_h)
    count = 0
    while cv.waitKey(1) < 0:
        print(args.input)
        has_frame, img = cap.read() 
        img = cv.resize(img, dim)  
        #print(args.input)
        height, width, ch = img.shape    
        
        yunet.setInputSize([width, height])
        faces = yunet.detect(img) # # faces: None, or nx15 np.array
        try:
            face_align = recognizer.alignCrop(img, faces[1][0])
        except:
            print('error')
        
        img_name = args.input.split('.')[0] + "_" + str(count) + ".jpg"
        count += 1
        #print(img)
        cv.imwrite(target + img_name, face_align)
        #cv.imwrite(target + "ori" + img_name, img)
        cv.imshow('frame', face_align)
        #visualize(img, faces, img_name)   
        #cv.imshow('frame', img)         
     
cv.destroyAllWindows()