import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
import os


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(200, 200, 300, 300)
        self.setWindowTitle("Create folder dialog")
        self.initUI()

    def initUI(self):
        # Create a push button
        btn = QPushButton("Create a folder", self)
        btn.clicked.connect(self.createDialog)
        btn.resize(btn.sizeHint())
        btn.move(50, 50)

    def createDialog(self):
        # Create a folder dialog
        fdialog = QFileDialog()
        fdialog.setFileMode(QFileDialog.Directory)
        fdialog.setOption(QFileDialog.ShowDirsOnly, True)
        fdialog.setOption(QFileDialog.DontUseNativeDialog, True)
        if fdialog.exec_():
            directory = fdialog.selectedFiles()[0]
            if not os.path.exists(directory):
                os.makedirs(directory)


app = QApplication(sys.argv)
win = MyWindow()
win.show()
sys.exit(app.exec_())