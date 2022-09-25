from PyQt6.QtWidgets import QMainWindow, QApplication
from PyQt6 import uic, QtGui
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
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

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
            qt_img = self.convert_cv_qt(cv_img)   

            self.label.resize(cv_img.shape[1], cv_img.shape[0])
            self.label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):    
            scale_percent = 30
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
