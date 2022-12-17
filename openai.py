# create gui with pyqt and use opencv to play video and add a trackbar to control frame position

import cv2
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider
from PyQt5.QtCore import Qt, QTimer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.updating = True
        self.initUI()

    def initUI(self):
        # Create a label to display the video frames
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 640, 480)

        # Load the video
        self.capture = cv2.VideoCapture('video.mp4')
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a trackbar to control the frame position
        self.trackbar = QSlider(Qt.Horizontal, self)
        self.trackbar.setRange(0, self.frame_count)
        self.trackbar.valueChanged.connect(self.updateFrame)

        # Set up the timer to update the video frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(int(1000 / self.fps))

        # Show the window
        self.show()

    def updateFrame(self, source=None):
        # Get the current trackbar position
        frame_idx = self.trackbar.value()

        # Set the video to the corresponding frame
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame from the video
        success, frame = self.capture.read()

        if success:
            # Convert the frame to a QImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)

            # Update the label with the new frame
            self.label.setPixmap(QtGui.QPixmap.fromImage(image))

            # Update the trackbar position if the signal came from the trackbar
            if source == self.trackbar:
                self.trackbar.setValue(frame_idx + 1)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec_()