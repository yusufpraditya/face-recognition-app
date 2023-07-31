from PyQt6.QtWidgets import QMainWindow, QApplication, QSizePolicy
from PyQt6 import uic, QtGui
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtMultimedia import *
from PyQt6.QtMultimediaWidgets import *
import cv2
import numpy as np
import sys

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)

class MyGUI(QMainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi("form.ui", self)
        self.show()
        self.cameraList.addItem("camera 1")
        self.cameraList.addItem("camera 2")
        #self.label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.frame1.setStyleSheet(
			"color: rgb(255,0,255);"
			"background-color: rgb(0,0,0);"
			"qproperty-alignment: AlignCenter;")
        self.frame2.setStyleSheet(
			"color: rgb(255,0,255);"
			"background-color: rgb(0,0,0);"
			"qproperty-alignment: AlignCenter;")
        myCamera = QCamera()
        cameraDevice = myCamera.cameraDevice()
        cameraDescription = cameraDevice.description()
        self.cameraList.addItem(cameraDescription)
        self.testLabel.setFont(QFont("Times", 30))
        self.testLabel.adjustSize()

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
            qt_img = self.convert_to_qt(cv_img)
            self.frame1.resize(cv_img.shape[1], cv_img.shape[0])            
            self.frame1.adjustSize()

            self.frame2.resize(int(cv_img.shape[1]/2), int(cv_img.shape[0]/2))            
            #self.frame2.adjustSize()

            self.testLabel.setText(str(self.cameraList.currentIndex()))
            print(self.cameraList.currentIndex())
            
            self.frame1.setPixmap(qt_img)
            self.frame2.setPixmap(qt_img)
            

    def convert_to_qt(self, cv_img):    
            scale_percent = 100
            
            height = int(cv_img.shape[0] * (scale_percent / 100))
            width = int(cv_img.shape[1] * (scale_percent / 100))
            dim = (width, height)
            cv_img = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
            h, w, ch = cv_img.shape
            stride = cv_img.strides[0]
            convert_to_Qt_format = QtGui.QImage(cv_img, w, h, stride, QtGui.QImage.Format.Format_BGR888)            
            return QPixmap.fromImage(convert_to_Qt_format)

def main():
    app = QApplication([])
    window = MyGUI()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()