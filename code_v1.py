from shutil import ExecError
from PyQt6.QtWidgets import QMainWindow, QApplication, QSizePolicy, QLabel, QFileDialog, QMessageBox
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

    global cameraIndex, file_model_deteksi, file_model_pengenalan

    def run(self):        
        if cameraIndex == 0:
            cap = cv2.VideoCapture(cameraIndex)
        if cameraIndex == 1:
            cap = cv2.VideoCapture(cameraIndex,cv2.CAP_DSHOW)

        self.isActive = True

        while self.isActive:
            _, original_img = cap.read()
            
            model = YuNet(model_path=file_model_deteksi)
            
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

    def stop(self):
        self.isActive = False
        self.quit()
            
class MyGUI(QMainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi("desain_v3.ui", self)       
        self.show()
        
        self.display_width = 100
        self.display_height = 100
        
        self.pilihanTab.setEnabled(False)
        self.btnPauseRegistrasi.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(False)

        # Pilih tab registrasi/pengenalan
        self.btnRegistrasi.clicked.connect(self.tab_registrasi)
        self.btnPengenalan.clicked.connect(self.tab_pengenalan)

        # Dialog file model deteksi wajah
        self.btnModelDeteksi.clicked.connect(self.dialog_deteksi_wajah)

        # Dialog file model pengenalan wajah
        self.btnModelPengenalan.clicked.connect(self.dialog_pengenalan_wajah)        

        # Pilih input
        self.btnKameraRegistrasi.clicked.connect(self.box_kamera_registrasi)
        self.btnFotoRegistrasi.clicked.connect(self.line_foto_registrasi)       

        # Lokasi penyimpanan gambar wajah
        self.btnSimpanWajah.clicked.connect(self.dialog_folder_wajah)

        # Edit nama wajah
        self.btnNamaWajah.clicked.connect(self.nama_wajah)

        # Tombol
        self.btnStartRegistrasi.clicked.connect(self.tombol_start)
        self.btnPauseRegistrasi.clicked.connect(self.tombol_pause)
        

        self.crop.setMaximumHeight(self.crop.height())
        self.crop.setMaximumWidth(self.crop.width())
        self.align.setMaximumHeight(self.align.height())
        self.align.setMaximumWidth(self.align.width())

        self.thread = VideoThread()
        self.thread.detection_signal.connect(self.update_detection)
        self.thread.crop_signal.connect(self.update_crop)
        self.thread.alignment_signal.connect(self.update_align)

    def dialog_deteksi_wajah(self):
        global file_model_deteksi
        file = QFileDialog.getOpenFileName(self, "Masukkan file model deteksi wajah", "", "ONNX File (*.onnx)")
        if file:
            file_model_deteksi = str(file[0])
            self.lnDeteksi.setText(file_model_deteksi)
            print(file_model_deteksi)
    
    def dialog_pengenalan_wajah(self):
        global file_model_pengenalan
        file = QFileDialog.getOpenFileName(self, "Masukkan file model pengenalan wajah", "", "ONNX File (*.onnx)")
        if file:
            file_model_pengenalan = str(file[0])
            self.lnPengenalan.setText(file_model_pengenalan)
            print(file_model_pengenalan)
    
    def tab_registrasi(self):
        self.pilihanTab.setEnabled(True)
        self.pilihanTab.setCurrentIndex(0)
    
    def tab_pengenalan(self):
        self.pilihanTab.setEnabled(True)
        self.pilihanTab.setCurrentIndex(1)
    
    def box_kamera_registrasi(self):
        self.boxKameraRegistrasi.setEnabled(True)
        self.lnFotoRegistrasi.setEnabled(False)
        self.btnLokasiFoto.setEnabled(False)
        self.boxKameraRegistrasi.clear()

        # Tambah list kamera ke combobox
        cameraList = QMediaDevices.videoInputs()        
        for c in cameraList:
            self.boxKameraRegistrasi.addItem(c.description())
    
    def line_foto_registrasi(self):
        self.boxKameraRegistrasi.setEnabled(False)
        self.lnFotoRegistrasi.setEnabled(True)
        self.btnLokasiFoto.setEnabled(True)

    def dialog_folder_wajah(self):
        direktori = QFileDialog.getExistingDirectory(self, "Pilih folder penyimpanan wajah")
        if direktori:
            self.lnLokasi.setText(str(direktori))
    
    def nama_wajah(self):
        if self.btnNamaWajah.text() == "Terapkan":
            self.btnNamaWajah.setText("Ganti")            
            self.lnNamaWajah.setEnabled(False)
        else:
            self.btnNamaWajah.setText("Terapkan")
            self.lnNamaWajah.setEnabled(True)
    
    def tombol_start(self):
        global cameraIndex
        if self.lnDeteksi.text() == "" or self.lnPengenalan.text() == "":
            QMessageBox.information(None, "Error", "Mohon masukkan file model deteksi & pengenalan pada bagian Setting.")
        else:
            cameraIndex = self.boxKameraRegistrasi.currentIndex()
            self.btnStartRegistrasi.setEnabled(False)      
            self.btnPauseRegistrasi.setEnabled(True)          
            self.thread.start()
        
    
    def tombol_pause(self):
        self.btnStartRegistrasi.setEnabled(True)
        self.btnPauseRegistrasi.setEnabled(False)
        self.thread.stop()

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