import imp
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Create the push button and label
        self.button = QPushButton('Start/Stop')
        self.label = QLabel()

        # Create the layout
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Create the camera and viewfinder objects
        self.camera = QCamera()
        self.viewfinder = QCameraViewfinder()
        self.camera.setViewfinder(self.viewfinder)

        # Connect the button's clicked signal to the slot function
        self.button.clicked.connect(self.start_stop_capture)

    def start_stop_capture(self):
        if self.camera.state() == QCamera.ActiveState:
            # Stop the camera
            self.camera.stop()
        else:
            # Start the camera
            self.camera.start()

        # Use the viewfinder's signal/slot mechanism to display the video frames
        self.viewfinder.frameAvailable.connect(self.update_frame)

    def update_frame(self):
        # Get the frame from the viewfinder and set it as the pixmap of the label
        frame = self.viewfinder.getFrame()
        self.label.setPixmap(frame)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())