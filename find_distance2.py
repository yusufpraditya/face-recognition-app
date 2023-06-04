import cv2
import numpy as np

known_distance = 25
known_width = 14

font = cv2.FONT_HERSHEY_SIMPLEX

def focal_length_finder(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length

def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance

def inference(frame, yunet):
    faces = yunet.detect(frame)
    return faces[1]

def get_width(result):
    face_width = 0
    for _, face in enumerate(result):
        x, y, w, h = np.maximum(face[0:4].astype(np.int32), 0)
        face_width = w    
    return face_width

def img_face_data(image):
    height, width, c = image.shape

    yunet = cv2.FaceDetectorYN.create(
        model="face_detection_yunet.onnx",
        config="",
        input_size=(width, height),
        score_threshold=0.9,
        nms_threshold=0.3,
        top_k=100)
    
    result = inference(image, yunet)
    
    if result is not None:
        width = get_width(result)
        return width

def visualize(frame, result):
    output = frame.copy()
    for _, face in enumerate(result):
        x, y, w, h = np.maximum(face[0:4].astype(np.int32), 0)
        start_point = (x, y)
        end_point = (x + w, y + h)
        rectangle_color = (0, 255, 0)
        cv2.rectangle(output, start_point, end_point, rectangle_color, thickness=2)
    return output

def main():
    ref_image = cv2.imread("saved_frame/face1.png")
    ref_image_face_width = img_face_data(ref_image)
    focal_length_found = focal_length_finder(known_distance, known_width, ref_image_face_width)
    print(ref_image_face_width)
    print(focal_length_found)

    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    yunet = cv2.FaceDetectorYN.create(
        model="face_detection_yunet.onnx",
        config="",
        input_size=(width, height),
        score_threshold=0.7,
        nms_threshold=0.3,
        top_k=100)

    while cv2.waitKey(1) < 0:
        _, frame = cap.read()

        result = inference(frame, yunet)
        
        if result is not None:
            face_width_in_frame = get_width(result)
            if face_width_in_frame != 0:
                distance = distance_finder(focal_length_found, known_width, face_width_in_frame)
                print(round(distance, 1), "cm", " - ", "score:", result[0][14])
            detected_face = visualize(frame, result)

            text = str(round(distance, 1)) + " cm - score: " + str(result[0][14])
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = int((detected_face.shape[1] - textsize[0]) / 2)
            textY = int((detected_face.shape[0] + textsize[1]) - 40)

            cv2.putText(detected_face, text, (textX, textY), font, 1, (0, 255, 0), 2)
            cv2.imshow("Detected Face", detected_face)

    cap.release()
    cv2.destroyAllWindows()

main()