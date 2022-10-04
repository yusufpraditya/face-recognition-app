from shutil import ExecError
from PyQt6.QtWidgets import QMainWindow, QApplication, QSizePolicy
from PyQt6 import uic, QtGui
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtMultimedia import *
from PyQt6.QtMultimediaWidgets import *
import cv2
from more_itertools import peekable
import numpy as np
import sys
from yunet import YuNet

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_pixmap_signal2 = pyqtSignal(np.ndarray)
    change_pixmap_signal3 = pyqtSignal(np.ndarray)
    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, original_img = cap.read()
            model = YuNet(model_path="face_detection_yunet.onnx")
            
            detected_img, landmarks = model.detect(original_img)  
            if detected_img is not None:
                self.change_pixmap_signal.emit(detected_img)
            else:
                self.change_pixmap_signal.emit(original_img)
            
        

                    
            
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

        # Get the resize Event Callback
        self.resizeEvent = self.label_resize
        self.detection.resizeEvent = self.camera_resize

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_pixmap_signal2.connect(self.update_crop)
        self.thread.change_pixmap_signal3.connect(self.update_align)
        self.thread.start()

    # Resize Event Callback
    def label_resize(self, resizeEvent:QResizeEvent):
        self.detection.resize(resizeEvent.size())
        print("here")

    # Resize Event Callback
    def camera_resize(self, resizeEvent:QResizeEvent):
        self.display_width, self.display_height = self.detection.width(), self.detection.height()
        print(self.detection.width())  

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
            qt_img = self.convert_to_qt(cv_img)    
            self.detection.adjustSize()                         
            self.detection.setPixmap(qt_img)
            self.detection.setScaledContents(True)        
    
    @pyqtSlot(np.ndarray)
    def update_crop(self, cropped_img):
        qt_img2 = self.convert_to_qt(cropped_img)
        self.crop.adjustSize()
        self.crop.setPixmap(qt_img2)
        self.crop.setScaledContents(True)  

    @pyqtSlot(np.ndarray)
    def update_align(self, aligned_face):
        qt_img3 = self.convert_to_qt(aligned_face)
        self.align.adjustSize()
        self.align.setPixmap(qt_img3)
        self.align.setScaledContents(True)  
            

    def convert_to_qt(self, cv_img):    
            scale_percent = 100
            
            height = int(cv_img.shape[0] * (scale_percent / 100))
            width = int(cv_img.shape[1] * (scale_percent / 100))
            dim = (width, height)
            cv_img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
            h, w, ch = cv_img.shape
            stride = cv_img.strides[0]
            
            convert_to_Qt_format = QtGui.QImage(cv_img, w, h, stride, QtGui.QImage.Format.Format_BGR888)    
            #duplication = convert_to_Qt_format.copy()
            p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)   
            
            return QPixmap.fromImage(convert_to_Qt_format)

def main():
    app = QApplication([])
    window = MyGUI()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()