import cv2
import numpy as np

def inference(frame, yunet):
    faces = yunet.detect(frame)
    return faces[1]

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
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    yunet = cv2.FaceDetectorYN.create(
    model="face_detection_yunet.onnx",
    config="",
    input_size=(width, height),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=100)

    img_counter = 0

    while cv2.waitKey(1) < 0:
        _, frame = cap.read()

        result = inference(frame, yunet)

        k = cv2.waitKey(1)
        if k%256 == 32:
            img_name = "saved_frame/opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

        if result is not None:
            detected_face = visualize(frame, result)

            cv2.imshow("Detected Face", detected_face)

    cap.release()
    cv2.destroyAllWindows()

main()