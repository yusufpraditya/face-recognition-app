from PyQt6 import QtCore, QtGui, QtWidgets
import cv2
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from numpy import ndarray



class Camera(QWidget):
    def __init__(self, port):
        super().__init__()

        self.display_width = 640
        self.display_height = 480


        self.camera = QLabel()
        # self.camera.setGeometry(0,0,640,480)
        self.camera.resize(self.display_width, self.display_height)
        
        #  Set Minimum size of camera stream to avoid going in recursive loop
        self.camera.setMinimumWidth(640)
        self.camera.setMinimumHeight(480)

        self.label = QLabel(f'Port {port}')

        self.label.setStyleSheet("""
                color: black;
                font: 30px;
        """)

        # Layout

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.camera)

        self.setLayout(self.layout)

        # Get the resize Event Callback
        self.resizeEvent = self.label_resize
        self.camera.resizeEvent = self.camera_resize

        self.thread = VideoThread(port)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    # Resize Event Callback
    def label_resize(self, resizeEvent:QResizeEvent):
        self.camera.resize(resizeEvent.size())

    # Resize Event Callback
    def camera_resize(self, resizeEvent:QResizeEvent):
        self.display_width, self.display_height = self.camera.width(), self.camera.height()


    def close_event(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.camera.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(ndarray)

    def __init__(self, port):
        super().__init__()
        self.running = True
        self.port = port

    def run(self):
        self.capture = cv2.VideoCapture(self.port)

        while self.running:
            self.ret, self.image = self.capture.read()
            if self.ret:
                self.change_pixmap_signal.emit(self.image)

        self.capture.release()

    def stop(self):
        self.running = False
        self.wait()