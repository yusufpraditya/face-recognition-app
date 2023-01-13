import argparse
import numpy as np
import cv2 as cv
import os
import pickle
from serial.tools.list_ports import comports
import serial
import time
import datetime


tm = cv.TickMeter()

# default: 0.363 and 1.128
# higher cosine, better result
# lower l2, better result
cosine_similarity_threshold = 0.463
l2_similarity_threshold = 1.0

ser = []

global connected
connected = False

def mode(array):
    if array == []:
        return 'unknown'
    most = max(list(map(array.count, array)))
    list_mode = list(set(filter(lambda x: array.count(x) == most, array)))
    if len(list_mode) > 1:
        return 'unknown'
    else:
        return list_mode[0]

def cam_index():
    webcam_index = 0
    arr_webcam = []
    while True:        
        c = cv.VideoCapture(webcam_index)
        
        if not c.read()[0]:
            webcam_index += 1
            if webcam_index == 11:
                webcam_index = 0
        else:
            arr_webcam.append(webcam_index)
            break
        c.release()
    print("Cam Index: " + str(arr_webcam))
    return arr_webcam

def connect():      
    ser.clear()
    ports_name = []
    ports = [p.name for p in comports()]
    print("len ports: " + str(len(ports)))
    for i in range(len(ports)):
        if len(ports) > 0:
            if "USB" in ports[i] or "ACM" in ports[i]:
                print(ports[i])
                ports_name.append("/dev/" + ports[i])
                print(ports_name)
                ser.append(serial.Serial(ports_name[i], timeout=0))
                print(ser)    

def find_face(input, faces, thickness=2):      
    face_img = input.copy()   
    for idx, face in enumerate(faces[1]):
        coords = face[:-1].astype(np.int32)
        for i in range(len(coords)):
            if coords[i] < 0:
                coords[i] = 0
        cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
        face_img = face_img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]                
        if len(face_img) != 0:                
            return face_img    

def write_serial(identity):
    if ser == []:
        connect()
    print(ser)
    try:
        if identity == "unknown":            
            for i in range(len(ser)):
                print("unknown...")
                data_str = identity + ",0\n"
                ser[i].write(str.encode(data_str))
        else:        
            for i in range(len(ser)):
                print("written to serial...")   
                data_str = identity + ",1\n"             
                ser[i].write(str.encode(data_str))
    except:
        connect()          

def recognize(img, yunet_detect, database):
    recognizer = cv.FaceRecognizerSF.create("face_recognition_sface.onnx", "")        
    face_align = recognizer.alignCrop(img, yunet_detect[1][0])   
    face_feature = recognizer.feature(face_align)
    
    max_cosine = 0
    min_l2 = 2
    identity = 'unknown'

    for key in database.keys():
        name = key.split("_")[0]        
        if name != "img":
            cosine_score = recognizer.match(face_feature, database[key], cv.FaceRecognizerSF_FR_COSINE)
            if cosine_score > max_cosine:
                max_cosine = cosine_score
                identity = key
    if max_cosine >= cosine_similarity_threshold:
        identity = identity.split("_")[0]
    else:
        identity = "unknown"

    return identity, max_cosine

def setup():
    pickle_database = open("data.pkl", "rb")
    database = pickle.load(pickle_database)
    print(database.keys())
    pickle_database.close()
    print("DONE SETUP")
    return database

def loop(database):
    face_detected = 0    
    unknown_wait_time = 5
    known_wait_time = 15
    start_time = time.time()
    wait_time = time.time()
    is_waiting = False
    write_count = 0
    arr_identity = []
    arr_identity_serial = []
    yunet = cv.FaceDetectorYN.create(
        model="face_detection_yunet.onnx",
        config='',
        input_size=(320, 320),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=5000        
    )
    webcam_index = cam_index()    
    #cap = cv.VideoCapture(video_name)
    cap = cv.VideoCapture(webcam_index[0])
    scale_percent = 30
    frame_w = int(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) * scale_percent / 100)
    frame_h = int(int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) * scale_percent / 100)    
    
    dim = (frame_w, frame_h)
    while cv.waitKey(1) < 0:      
        tm.start()        
        has_frame, img = cap.read()
        if has_frame:
            img = cv.resize(img, dim)
            height, width, ch = img.shape
            yunet.setInputSize([width, height])
            yunet_detect = yunet.detect(img)  
            if yunet_detect[1] is not None:
                face_detected += 1                 
                #face_img = find_face(img, yunet_detect)
                
                identity, cosine_score = recognize(img, yunet_detect, database) 
                arr_identity.append(identity)

                #print("face name: ", identity)    
                #print("cosine score: ", cosine_score)
                #print(" ")
            else:
                face_detected = 0
                start_time = time.time()
                arr_identity.clear()
            #print("face detected count: " + str(face_detected))
            if face_detected > 8:                
                identity = mode(arr_identity)
                mean_identity = arr_identity.count(identity) / len(arr_identity)
                if mean_identity < 0.6:
                    identity = 'unknown'
                elapsed_time = time.time() - start_time
                print(arr_identity, identity, mean_identity, round(elapsed_time, 2))
                # if write_count == 0:
                #     print(write_count)
                arr_identity_serial.append(identity)
                if arr_identity_serial.count(identity) == 1:
                    is_waiting = False
                    print("written 1 identity")
                    write_serial(identity)     
                    if len(arr_identity_serial) > 1:
                        arr_identity_serial.clear()
                        arr_identity_serial.append(identity)
                if arr_identity_serial.count(identity) == 2:           
                    is_waiting = True
                    wait_time = time.time()         
                if is_waiting:
                    elapsed_wait_time = time.time() - wait_time                    
                    if identity == 'unknown':
                        if elapsed_wait_time > unknown_wait_time:
                            is_waiting = False
                            arr_identity_serial.clear()
                        else:
                            print("tunggu " + str(round(unknown_wait_time - elapsed_wait_time, 2)) + " detik")
                    else:
                        if elapsed_wait_time > known_wait_time:
                            is_waiting = False
                            arr_identity_serial.clear()
                        else:
                            print("tunggu " + str(round(known_wait_time - elapsed_wait_time, 2)) + " detik")
                arr_identity.clear()
                write_count += 1
                face_detected = 0
                start_time = time.time()
                tm.stop()                
            else:
                write_count = 0
            serial_written = False
            cv.imshow('frame', img)    
   

        else:
            #txt_file.close()
            #break
            print('waiting for webcam...')
            webcam_index = cam_index()    
            cap = cv.VideoCapture(webcam_index[0])
            time.sleep(1)

if __name__ == '__main__':   
    connect()
    database = setup()
    loop(database)
