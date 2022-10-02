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
# Inference parameters
parser.add_argument('--score_threshold', default=0.9, type=float, help='Threshold for filtering out faces with conf < conf_thresh.')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='Threshold for non-max suppression.')
parser.add_argument('--top_k', default=5000, type=int, help='Keep keep_top_k for results outputing.')
args = parser.parse_args()

now = datetime.datetime.now()
folder_name = now.strftime(args.input.split('.')[0] + "_%H%M%S/")
target = os.getcwd() + '/saved_yunet/' + folder_name

os.makedirs(target)

def visualize(input, faces, img_name, thickness=2):     
    face_img = input.copy()  
    print(target + img_name)
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            face_img = face_img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
            cv.imshow('detected', input)  
            if len(face_img) != 0:
                #print(len(face_img))
                #print(face_img)
                try:
                    cv.imwrite(target + img_name, face_img)
                except:
                    print(len(face_img))
                    #print(face_img)
            #cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            #cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            #cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            #cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            #cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    

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
    
    #frame_w = 640
    #frame_h = 360
    
    if args.input == "0":
        cap = cv.VideoCapture(int(args.input))
    else:
        cap = cv.VideoCapture(args.input)
    scale_percent = 30 # percent of original size
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
        
        img_name = args.input.split('.')[0] + "_" + str(count) + ".jpg"
        count += 1
        visualize(img, faces, img_name)   
        #cv.imshow('frame', img)         
     
cv.destroyAllWindows()