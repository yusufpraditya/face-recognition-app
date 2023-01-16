from shutil import ExecError
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5 import uic, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import cv2
import numpy as np
import sys
import os
import pickle
import datetime
import time
import subprocess

class VideoThread(QThread):    
    face_signal = pyqtSignal(np.ndarray)    
    db_face_signal = pyqtSignal(np.ndarray)
    webcam_error_signal = pyqtSignal(str)
    
    def __init__(self, my_gui):
        super().__init__()
        self.isActive = True
        self.isStopped = False
        self.my_gui = my_gui

    def run(self):     
        global aligned_img  
        cap = cv2.VideoCapture(self.my_gui.cameraIndex) 
        
        print("camera index: ", self.my_gui.cameraIndex)
        scale_percent = 30
        
        frame_w = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale_percent / 100)
        frame_h = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale_percent / 100)   
        dim = (frame_w, frame_h)
        
        tm = cv2.TickMeter()

        if not cap.isOpened():            
            self.webcam_error_signal.emit('Webcam sedang digunakan pada program lain. Tutup terlebih dahulu program tersebut.')

        while self.isActive: 
            tm.start()            
            if self.my_gui.pause:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.my_gui.frame_video)
                self.my_gui.pause = False
            has_frame, original_img = cap.read()         
            if has_frame:
                original_img = cv2.resize(original_img, dim)
                model_yunet = cv2.FaceDetectorYN.create(
                    model=self.my_gui.file_model_deteksi,
                    config='',
                    input_size=(320, 320),
                    score_threshold=0.7,
                    nms_threshold=0.3,
                    top_k=5000        
                )
                
                height, width, ch = original_img.shape
                model_yunet.setInputSize([width, height])

                model_sface = cv2.FaceRecognizerSF.create(self.my_gui.file_model_pengenalan, "")   
                
                detected_img = model_yunet.detect(original_img)
                
                try:
                    aligned_img = model_sface.alignCrop(original_img, detected_img[1][0])       
                    face_feature = model_sface.feature(aligned_img)
                    if detected_img is not None and aligned_img is not None and face_feature is not None:                        
                        self.face_signal.emit(aligned_img)
                    else:
                        self.my_gui.clear_label()
                    tm.stop()
                    fps = "{:.2f}".format(round(tm.getFPS(), 2))
                    print(fps)
                except:
                    self.my_gui.clear_label()
            if self.isStopped:                
                print("program stopped")
                self.my_gui.clear_label()
                self.isStopped = False
                self.isActive = False 
                break
    def stop(self):     
        self.quit()
        self.isActive = False
            
class MyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("register.ui", self)
        self.show()

        self.file_model_deteksi = "face_detection_yunet.onnx"
        self.file_model_pengenalan = "face_recognition_sface.onnx"
        self.cameraIndex = None
        self.pause = False
        self.stop = False
        self.database_kosong = False
        self.path_video = ""
        self.database_keys = []
        self.database_index = 0
        self.panjang_frame_video = 0
        self.frame_video = 0

        self.display_width = 100
        self.display_height = 100

        self.face.setMaximumSize(self.face.width(), self.face.height())
        self.dbFace.setMaximumSize(self.dbFace.width(), self.dbFace.height())
        
        self.pilihanTab.setCurrentIndex(0)

        self.pilihanTab.setEnabled(False)
        self.btnPauseRegistrasi.setEnabled(False)
        self.btnRegister.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(False)        

        # Pilih tab registrasi/pengenalan/edit database
        self.btnRegistrasi.clicked.connect(self.tab_registrasi)
        self.btnEditDB.clicked.connect(self.tab_edit_database) 

        # Edit nama wajah
        self.btnNamaWajah.clicked.connect(self.nama_wajah)
        self.btnEditNama.clicked.connect(self.edit_nama_wajah)
        self.btnBatalEdit.clicked.connect(self.batal_edit_nama)

        # Tombol-tombol Tab Registrasi
        self.btnStartRegistrasi.clicked.connect(self.tombol_start)
        self.btnPauseRegistrasi.clicked.connect(self.tombol_pause)
        self.btnStopRegistrasi.clicked.connect(self.tombol_stop)
        self.btnRegister.clicked.connect(self.tombol_register)    
        self.btnRefresh.clicked.connect(self.tombol_refresh)

        # Tombol-tombol Tab Edit Database
        self.btnNextFrame.clicked.connect(self.tombol_next_frame)
        self.btnPrevFrame.clicked.connect(self.tombol_prev_frame)
        self.btnHapusFrame.clicked.connect(self.tombol_hapus_frame)

        # Tombol keluar
        self.btnExit.clicked.connect(self.tombol_exit)
        

        self.thread = VideoThread(self)        
        self.thread.face_signal.connect(self.update_face)        
        self.thread.db_face_signal.connect(self.update_db_face)
        self.thread.webcam_error_signal.connect(self.show_message)    

    def tab_registrasi(self):
        self.clear_label()
        self.update_list_database()
        self.lnNamaWajah.clear()
        self.lnNamaWajah.setEnabled(True)
        self.btnNamaWajah.setText("Terapkan")
        
        self.pilihanTab.setEnabled(True)
        self.pilihanTab.setCurrentIndex(0)        

        self.btnPauseRegistrasi.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(False)

        self.boxKameraRegistrasi.setEnabled(True)        
        self.btnStartRegistrasi.setEnabled(True)

        # Tambah list kamera ke combobox
        self.boxKameraRegistrasi.clear()
        cameraList = QCameraInfo.availableCameras()  
        for c in cameraList:
            self.boxKameraRegistrasi.addItem(c.description())
    
    def tab_edit_database(self):
        self.clear_label()
        self.lnEditNama.clear()
        self.lnEditNama.setEnabled(False)
        self.btnEditNama.setText("Ganti")

        self.pilihanTab.setEnabled(True)
        self.pilihanTab.setCurrentIndex(1)

        self.btnPrevFrame.setEnabled(False)
        self.btnNextFrame.setEnabled(False)
        self.btnHapusFrame.setEnabled(False)        

        pickle_database = open("data.pkl", "rb")
        database = pickle.load(pickle_database)
        pickle_database.close()    
        self.update_list_database()

        if database != {}:      
            self.database_kosong = False   
            if self.database_keys == []:   
                self.btnNextFrame.setEnabled(True)
                self.btnPrevFrame.setEnabled(True)
                self.btnHapusFrame.setEnabled(True)
                for key in database.keys():                    
                    if "img" in key:
                        self.database_keys.append(key)
                print(self.database_keys)                
                self.update_db_face(database[self.database_keys[0]])
                self.display_nama_wajah(self.database_keys[self.database_index])  
            else:
                self.btnNextFrame.setEnabled(True)
                self.btnPrevFrame.setEnabled(True)
                self.btnHapusFrame.setEnabled(True)
                self.update_db_face(database[self.database_keys[0]])
                self.display_nama_wajah(self.database_keys[self.database_index])  
        else:
            self.database_kosong = True
            QMessageBox.information(None, "Error", "Database kosong. Tambahkan database wajah pada menu registrasi wajah.")   
            
    def update_list_database(self):
        self.listDatabase.clear()
        pickle_database = open("data.pkl", "rb")
        database = pickle.load(pickle_database)
        
        pickle_database.close()
        name_counts = {}
        for key in database.keys():                
            name = key.split("_")[0]
            if name == "img":
                pass
            else:
                if name not in name_counts:
                    name_counts[name] = 0
                name_counts[name] += 1        
        for name, count in name_counts.items():
            str_list = name + " (" + str(count) + " Frame)"    
            self.listDatabase.addItem(str_list)

    def nama_wajah(self):
        if self.btnNamaWajah.text() == "Terapkan":
            if self.lnNamaWajah.text() == "":
                QMessageBox.information(None, "Error", "Nama wajah tidak boleh kosong!")
            elif "_" in self.lnNamaWajah.text():
                QMessageBox.information(None, "Error", 'Tidak dapat menggunakan simbol "_"')
            elif len(self.lnNamaWajah.text()) > 17:
                QMessageBox.information(None, "Error", 'Nama yang dimasukkan terlalu panjang (maksimal 17 karakter).')
            else:
                self.btnNamaWajah.setText("Ganti")
                self.lnNamaWajah.setEnabled(False)
                    
        else:
            self.btnNamaWajah.setText("Terapkan")
            self.lnNamaWajah.setEnabled(True)

    @pyqtSlot(str)
    def show_message(self, message):
        self.thread.stop()
        self.btnStartRegistrasi.setEnabled(True)      
        self.btnPauseRegistrasi.setEnabled(False)    
        self.btnRegister.setEnabled(False)      
        self.btnStopRegistrasi.setEnabled(False)   
        self.btnRegistrasi.setEnabled(True)
        self.btnEditDB.setEnabled(True)
        QMessageBox.information(None, "Error", message)

    def tombol_start(self):  
        self.thread.isStopped = False
        self.thread.isActive = True      
        for i in range(0, 11):
            self.cameraIndex = None
            webcam_path = '/sys/class/video4linux/video' + str(i) + '/name'
            print(webcam_path)
            cmd = ['cat', webcam_path]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            o, e = proc.communicate()            
            
            if proc.returncode == 0:      
                webcam_name = ' '.join(o.decode('ascii').split())          
                print(webcam_name)                
                print(self.boxKameraRegistrasi.currentText())
                if webcam_name == self.boxKameraRegistrasi.currentText():    
                    self.cameraIndex = i
                    print(self.cameraIndex)
                    break               
       
        if self.cameraIndex is not None:
            self.btnStartRegistrasi.setEnabled(False)      
            self.btnPauseRegistrasi.setEnabled(True)
            self.btnRegister.setEnabled(True)
            self.btnStopRegistrasi.setEnabled(True)
            self.btnRegistrasi.setEnabled(False)
            self.btnEditDB.setEnabled(False)
            self.thread.start()
        else:
            QMessageBox.information(None, "Error", "Webcam tidak tersambung.")
            
    def tombol_pause(self):
        self.pause = True
        self.stop = False
        self.btnStartRegistrasi.setEnabled(True)
        self.btnPauseRegistrasi.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(True)
        self.thread.stop()

    def tombol_stop(self):
        self.pause = False
        self.stop = True       
        
        self.thread.isStopped = True
        self.clear_label()
        self.btnStartRegistrasi.setEnabled(True)
        self.btnPauseRegistrasi.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(False)
        self.btnRegister.setEnabled(False)      
        self.btnRegistrasi.setEnabled(True)
        self.btnEditDB.setEnabled(True)
    
    def tombol_register(self):
        global aligned_img        

        if self.lnNamaWajah.text() == "":
            QMessageBox.information(None, "Error", "Mohon isi nama wajah yang akan diregistrasi.")
        elif self.btnNamaWajah.text() == "Terapkan":
            QMessageBox.information(None, "Error", "Klik tombol 'Terapkan' pada nama wajah terlebih dahulu.")
        else:             
            # Simpan gambar wajah ke folder database 
            duplikat = False
            now = datetime.datetime.now()
            time_now = now.strftime("%H%M%S")           
            
            # Simpan database dalam format pickle
            database = {}
            
            pickle_database = open("data.pkl", "rb")
            database = pickle.load(pickle_database)
            pickle_database.close()

            nama_file_gambar = "img_" + self.lnNamaWajah.text() + "_" + time_now
            database[nama_file_gambar] = aligned_img
            
            nama_wajah = self.lnNamaWajah.text() + "_" + time_now
            model_pengenalan = cv2.FaceRecognizerSF.create(self.file_model_pengenalan, "")

            if aligned_img is not None: 
                fitur_wajah = model_pengenalan.feature(aligned_img)

                for value in database.values():
                    if np.array_equal(value, fitur_wajah):
                        duplikat = True
                        break
                
                if duplikat:
                    QMessageBox.information(None, "Error", "Tidak dapat mendaftar wajah yang sudah dimasukkan sebelumnya.")
                    duplikat = False
                else:                    
                    database[nama_wajah] = fitur_wajah                    
                    pickle_database = open("data.pkl", "wb")
                    pickle.dump(database, pickle_database)
                    pickle_database.close()  
                    self.update_list_database()
                    QMessageBox.information(None, "Info", 'Wajah "' + self.lnNamaWajah.text() + '" berhasil ditambahkan.')
            else:
                QMessageBox.information(None, "Error", "Gagal.")
            pickle_database = open("data.pkl", "rb")
            database = pickle.load(pickle_database)
            pickle_database.close()

            name_counts = {}
            for key in database.keys():                
                name = key.split("_")[0]
                if name == "img":
                    pass
                else:
                    if name not in name_counts:
                        name_counts[name] = 0
                    name_counts[name] += 1

            self.listDatabase.clear()
            for name, count in name_counts.items():
                str_list = name + " (" + str(count) + " Frame)"                
                self.listDatabase.addItem(str_list)         

    def tombol_refresh(self):
        self.boxKameraRegistrasi.clear()
        cameraList = QCameraInfo.availableCameras()  
        for c in cameraList:
            self.boxKameraRegistrasi.addItem(c.description())
   
    def tombol_exit(self):
        sys.exit()
    
    def edit_nama_wajah(self):
        if self.btnEditNama.text() == "Terapkan":            
            if self.lnEditNama.text() == "":
                QMessageBox.information(None, "Error", "Nama wajah tidak boleh kosong!")
            else:
                if self.database_kosong == False:
                    if self.database_keys[self.database_index].split("_")[1] == self.lnEditNama.text():
                        QMessageBox.information(None, "Error", "Nama yang dimasukkan sama dengan sebelumnya.")
                    elif "_" in self.lnEditNama.text():
                        QMessageBox.information(None, "Error", 'Tidak dapat menggunakan simbol "_"')
                    elif len(self.lnEditNama.text()) > 17:
                        QMessageBox.information(None, "Error", 'Nama yang dimasukkan terlalu panjang (maksimal 17 karakter).')
                    else:
                        self.btnPrevFrame.setEnabled(True)
                        self.btnNextFrame.setEnabled(True)
                        self.btnHapusFrame.setEnabled(True)
                        self.btnEditNama.setText("Ganti")
                        self.lnEditNama.setEnabled(False)
                        self.btnBatalEdit.setEnabled(False)
                        
                        pickle_database = open("data.pkl", "rb")
                        database = pickle.load(pickle_database)
                        pickle_database.close()
                        new_img_name = "img_" + self.lnEditNama.text() + "_" + self.database_keys[self.database_index].split("_")[2]
                        new_name = self.lnEditNama.text() + "_" + self.database_keys[self.database_index].split("_")[2]

                        new_database_1 = dict((key.replace(self.database_keys[self.database_index], new_img_name), value) for key, value in database.items())
                        new_database_2 = dict((key.replace(self.database_keys[self.database_index].replace("img_", ""), new_name), value) for key, value in new_database_1.items())
                        
                        QMessageBox.information(None, "Info", 'Nama wajah "' + self.database_keys[self.database_index].split("_")[1] + '" berhasil diubah menjadi "' + self.lnEditNama.text() + '"')  

                        self.database_keys = [key.replace(self.database_keys[self.database_index], new_img_name) for key in self.database_keys]

                        print(new_database_2.keys())
                        print(self.database_keys)
                        
                        pickle_database = open("data.pkl", "wb")
                        pickle.dump(new_database_2, pickle_database)
                        pickle_database.close()
                        self.update_list_database()
                else:                   
                    QMessageBox.information(None, "Error", "Database kosong. Tambahkan database wajah pada menu registrasi wajah.")   

        else:
            self.btnPrevFrame.setEnabled(False)
            self.btnNextFrame.setEnabled(False)
            self.btnHapusFrame.setEnabled(False)
            self.btnEditNama.setText("Terapkan")
            self.lnEditNama.setEnabled(True)
            self.btnBatalEdit.setEnabled(True)

    def batal_edit_nama(self):
        self.display_nama_wajah(self.database_keys[self.database_index])
        self.btnPrevFrame.setEnabled(True)
        self.btnNextFrame.setEnabled(True)
        self.btnHapusFrame.setEnabled(True)
        self.btnEditNama.setText("Ganti")
        self.lnEditNama.setEnabled(False)
        self.btnBatalEdit.setEnabled(False)

    def tombol_next_frame(self):        
        pickle_database = open("data.pkl", "rb")
        database = pickle.load(pickle_database)
        pickle_database.close()
        self.database_index += 1
        if self.database_index < len(self.database_keys):
            self.update_db_face(database[self.database_keys[self.database_index]])
            print(self.database_keys[self.database_index])
            self.display_nama_wajah(self.database_keys[self.database_index])  
        else:
            self.database_index = 0
            self.update_db_face(database[self.database_keys[self.database_index]])
            print(self.database_keys[self.database_index])
            self.display_nama_wajah(self.database_keys[self.database_index])  
        print(self.database_index)
        print(" ")

    def tombol_prev_frame(self):        
        pickle_database = open("data.pkl", "rb")
        database = pickle.load(pickle_database)
        pickle_database.close()        
        self.database_index -= 1
        if self.database_index >= 0:
            self.update_db_face(database[self.database_keys[self.database_index]])
            print(self.database_keys[self.database_index])
            self.display_nama_wajah(self.database_keys[self.database_index])  
        else:
            self.database_index = len(self.database_keys) - 1
            self.update_db_face(database[self.database_keys[self.database_index]])
            print(self.database_keys[self.database_index])
            self.display_nama_wajah(self.database_keys[self.database_index])  
        print(self.database_index)
        print(" ")

    def tombol_hapus_frame(self):        
        pickle_database = open("data.pkl", "rb")
        database = pickle.load(pickle_database)
        pickle_database.close()  

        if self.database_keys != []:   
            print(self.database_keys[self.database_index])
            del database[self.database_keys[self.database_index]]
            del database[self.database_keys[self.database_index].replace("img_", "")]
            self.database_keys.remove(self.database_keys[self.database_index])

            if self.database_keys != []: 
                print(self.database_keys)
                print(self.database_index)
                print(" ")
                if self.database_index == 0:
                    self.update_db_face(database[self.database_keys[self.database_index]])
                    self.display_nama_wajah(self.database_keys[self.database_index])  
                else:
                    self.database_index -= 1
                    self.update_db_face(database[self.database_keys[self.database_index]])
                    self.display_nama_wajah(self.database_keys[self.database_index])                    
            else:
                self.dbFace.clear()
                self.lnEditNama.clear()
                self.listDatabase.clear()
                self.btnNextFrame.setEnabled(False)
                self.btnPrevFrame.setEnabled(False)
                self.btnHapusFrame.setEnabled(False)
                self.btnEditNama.setEnabled(False)
                self.dbFace.setText("DB Face")
                QMessageBox.information(None, "Error", "Semua data wajah sudah dihapus. Silakan tambahkan data baru melalui menu registrasi wajah.")
        else:
            self.dbFace.clear()
            self.dbFace.setText("DB Face")
            QMessageBox.information(None, "Error", "Data wajah kosong. Silakan tambahkan data baru melalui menu registrasi wajah.")
        
        pickle_database = open("data.pkl", "wb")
        pickle.dump(database, pickle_database)
        pickle_database.close()
        self.update_list_database()
        if self.lnEditNama.text() != "":
            QMessageBox.information(None, "Info", 'Wajah "' + self.lnEditNama.text() + '" berhasil dihapus.')
    
    def display_nama_wajah(self, nama):
        nama = nama.split("_")[1]
        self.lnEditNama.setText(nama)
        
    def clear_label(self):
        if self.pause:
            pass
        else:            
            print("label cleared")
            self.face.clear()
            self.face.setText("Face")            
            self.dbFace.clear()
            self.dbFace.setText("DB Face")      

    @pyqtSlot(np.ndarray)
    def update_face(self, face_img):
        h, w, _ = face_img.shape

        # Resize
        if h > self.face.height() or w > self.face.width():
           h_ratio = self.face.height() / h
           w_ratio = self.face.width() / w
           scale_factor = min(h_ratio, w_ratio)
           h = int(h * scale_factor)
           w = int(w * scale_factor)
           dim = (w, h)
           face_img = cv2.resize(face_img, dim)

        bytes_per_line = 3 * w
        qt_format = QtGui.QImage(face_img, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        qt_img = QPixmap.fromImage(qt_format)        
        self.face.setPixmap(qt_img)
   
    @pyqtSlot(np.ndarray)
    def update_db_face(self, face_img):
        h, w, _ = face_img.shape

        # Resize
        if h > self.dbFace.height() or w > self.dbFace.width():
           h_ratio = self.dbFace.height() / h
           w_ratio = self.dbFace.width() / w
           scale_factor = min(h_ratio, w_ratio)
           h = int(h * scale_factor)
           w = int(w * scale_factor)
           dim = (w, h)
           face_img = cv2.resize(face_img, dim)

        bytes_per_line = 3 * w
        qt_format = QtGui.QImage(face_img, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        qt_img = QPixmap.fromImage(qt_format)        
        self.dbFace.setPixmap(qt_img)   
    
def main():
    app = QApplication([])
    window = MyGUI()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
