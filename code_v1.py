from shutil import ExecError
from PyQt6.QtWidgets import QMainWindow, QApplication, QSizePolicy, QLabel
from PyQt6 import uic, QtGui
from PyQt6.QtGui import QPixmap, QResizeEvent, QBitmap
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtMultimedia import *
from PyQt6.QtMultimediaWidgets import *
import cv2
from more_itertools import peekable
import numpy as np
import sys
from yunet import YuNet

class VideoThread(QThread):
    #change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray)
    detection_signal = pyqtSignal(np.ndarray)
    crop_signal = pyqtSignal(np.ndarray)
    alignment_signal = pyqtSignal(np.ndarray)
    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, original_img = cap.read()
            
            model = YuNet(model_path="face_detection_yunet.onnx")
            
            detected_img, face_img, landmarks = model.detect(original_img)      
            try:       
                aligned_img = model.align_face(face_img, landmarks)
            except Exception as e:
                print(e)
            if detected_img is not None and landmarks is not None and aligned_img is not None:                
                self.detection_signal.emit(detected_img)
                self.crop_signal.emit(face_img)
                self.alignment_signal.emit(aligned_img)
            else:
                self.detection_signal.emit(original_img)

            
class MyGUI(QMainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi("desain_v1.ui", self)       
        self.show()

        self.display_width = 100
        self.display_height = 100

        myCamera = QCamera()
        cameraDevice = myCamera.cameraDevice()
        cameraDescription = cameraDevice.description()
        print(cameraDescription)

        self.crop.setMaximumHeight(self.crop.height())
        self.crop.setMaximumWidth(self.crop.width())
        self.align.setMaximumHeight(self.align.height())
        self.align.setMaximumWidth(self.align.width())

        self.detection.setStyleSheet(
			"color: rgb(255,0,255);"
			"background-color: rgb(0,0,0);"
			"qproperty-alignment: AlignCenter;")
        self.crop.setStyleSheet(
			"color: rgb(255,0,255);"
			"background-color: rgb(0,0,0);"
			"qproperty-alignment: AlignCenter;")
        self.align.setStyleSheet(
			"color: rgb(255,0,255);"
			"background-color: rgb(0,0,0);"
			"qproperty-alignment: AlignCenter;")
        self.img1.setStyleSheet(
			"color: rgb(255,0,255);"
			"background-color: rgb(0,0,0);"
			"qproperty-alignment: AlignCenter;")
        self.img2.setStyleSheet(
			"color: rgb(255,0,255);"
			"background-color: rgb(0,0,0);"
			"qproperty-alignment: AlignCenter;")

        self.thread = VideoThread()
        self.thread.detection_signal.connect(self.update_detection)
        self.thread.crop_signal.connect(self.update_crop)
        self.thread.alignment_signal.connect(self.update_align)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_detection(self, cv_img): 
        h, w, _ = cv_img.shape
        bytes_per_line = 3 * w
        qt_format = QtGui.QImage(cv_img, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)        
        qt_img = QPixmap.fromImage(qt_format)        
        self.detection.adjustSize()                    
        self.detection.setPixmap(qt_img)
        #self.detection.setScaledContents(True)

    @pyqtSlot(np.ndarray)
    def update_crop(self, face_img):
        face_img = face_img.copy()
        h, w, _ = face_img.shape
        bytes_per_line = 3 * w
        qt_format = QtGui.QImage(face_img, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)        
        qt_img = QPixmap.fromImage(qt_format)
        self.crop.adjustSize()
        self.crop.setPixmap(qt_img)
        #self.crop.setScaledContents(True)

    @pyqtSlot(np.ndarray)
    def update_align(self, face_img):        
        h, w, _ = face_img.shape
        bytes_per_line = 3 * w
        qt_format = QtGui.QImage(face_img, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        print(QtGui.QImage.Format.Format_BGR888)
        qt_img = QPixmap.fromImage(qt_format)
        self.align.adjustSize()
        self.align.setPixmap(qt_img)
        #self.align.setScaledContents(True)

def main():
    app = QApplication([])
    window = MyGUI()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()