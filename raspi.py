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
from yunet import YuNet
from sface import SFace
import datetime
import time

class VideoThread(QThread):    
    face_signal = pyqtSignal(np.ndarray)    
    db_face_signal = pyqtSignal(np.ndarray)

    global cameraIndex, lokasi_pickle
    
    def __init__(self, my_gui):
        super().__init__()
        self.my_gui = my_gui

    def run(self):     
        global aligned_img   
        if cameraIndex == 0:
            cap = cv2.VideoCapture(cameraIndex)
        if cameraIndex == 1:
            cap = cv2.VideoCapture(cameraIndex,cv2.CAP_DSHOW)
                    
        self.isActive = True
        tm = cv2.TickMeter()
        
        while self.isActive:
            tm.start()            
            if self.my_gui.pause:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.my_gui.frame_video)
                self.my_gui.pause = False
            _, original_img = cap.read()         

            model_yunet = YuNet(model_path=self.my_gui.file_model_deteksi)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            model_yunet.set_input_size([w, h])

            model_sface = SFace(model_path=self.my_gui.file_model_pengenalan)
            if original_img is not None:
                detected_img, face_img, landmarks = model_yunet.detect(original_img)
                aligned_img = model_yunet.align_face(face_img, landmarks)       
                face_feature = model_sface.feature(aligned_img)            
                if detected_img is not None and face_img is not None and landmarks is not None and aligned_img is not None and face_feature is not None:
                    self.face_signal.emit(aligned_img)     
                else:
                    self.my_gui.clear_label()
                tm.stop()
                fps = "{:.2f}".format(round(tm.getFPS(), 2))
                print(fps)

    def stop(self):      
        #time.sleep(1)      
        self.quit()
        self.isActive = False
        self.my_gui.clear_label()       
    
            
class MyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("raspi.ui", self)
        self.show()

        self.file_model_deteksi = "face_detection_yunet.onnx"
        self.file_model_pengenalan = "face_recognition_sface.onnx"
        
        self.pause = False
        self.stop = False
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

        # Lokasi penyimpanan gambar wajah
        self.btnSimpanWajah.clicked.connect(self.dialog_simpan_database)

        # Edit nama wajah
        self.btnNamaWajah.clicked.connect(self.nama_wajah)
        self.btnEditNama.clicked.connect(self.edit_nama_wajah)
        self.btnBatalEdit.clicked.connect(self.batal_edit_nama)

        # Tombol-tombol Tab Registrasi
        self.btnStartRegistrasi.clicked.connect(self.tombol_start)
        self.btnPauseRegistrasi.clicked.connect(self.tombol_pause)
        self.btnStopRegistrasi.clicked.connect(self.tombol_stop)
        self.btnRegister.clicked.connect(self.tombol_register)    

        # Lokasi file database yang akan diedit
        self.btnEditFileDB.clicked.connect(self.dialog_edit_database)

        # Tombol-tombol Tab Edit Database
        self.btnNextFrame.clicked.connect(self.tombol_next_frame)
        self.btnPrevFrame.clicked.connect(self.tombol_prev_frame)
        self.btnHapusFrame.clicked.connect(self.tombol_hapus_frame)

        # Tombol keluar
        self.btnExit.clicked.connect(self.tombol_exit)
        

        self.thread = VideoThread(self)        
        self.thread.face_signal.connect(self.update_face)        
        self.thread.db_face_signal.connect(self.update_db_face)

    def tab_registrasi(self):
        self.clear_label()
        self.listDatabase.clear()
        self.pilihanTab.setEnabled(True)
        self.pilihanTab.setCurrentIndex(0)        

        self.btnPauseRegistrasi.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(False)

        self.boxKameraRegistrasi.setEnabled(True)
        self.boxKameraRegistrasi.clear()
        self.btnStartRegistrasi.setEnabled(True)

        # Tambah list kamera ke combobox
        cameraList = QCameraInfo.availableCameras()  
        for c in cameraList:
            self.boxKameraRegistrasi.addItem(c.description())
    
    def tab_edit_database(self):
        self.clear_label()
        self.listDatabase.clear()
        self.pilihanTab.setEnabled(True)
        self.pilihanTab.setCurrentIndex(1)

        self.btnPrevFrame.setEnabled(False)
        self.btnNextFrame.setEnabled(False)
        self.btnHapusFrame.setEnabled(False)

    
    def dialog_simpan_database(self):
        file_database = QFileDialog.getSaveFileName(self, "Pilih lokasi penyimpanan database dan nama filenya", "", "Pickle File (*.pkl)")
        if file_database[0] != "":
            self.lnLokasiSimpanDB.setText(file_database[0])
            if os.path.isfile(file_database[0]):
                lokasi_pickle = file_database[0]
                pickle_database = open(lokasi_pickle, "rb")
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
            else:
                 self.listDatabase.clear()
    
    def nama_wajah(self):
        if self.btnNamaWajah.text() == "Terapkan":
            if self.lnNamaWajah.text() == "":
                QMessageBox.information(None, "Error", "Nama wajah tidak boleh kosong!")
            else:
                if self.lnLokasiSimpanDB.text() == "":
                    QMessageBox.information(None, "Error", "Pilih folder penyimpanan wajah terlebih dahulu!")
                else:
                    self.btnNamaWajah.setText("Ganti")
                    self.lnNamaWajah.setEnabled(False)
        else:
            self.btnNamaWajah.setText("Terapkan")
            self.lnNamaWajah.setEnabled(True)
    
    def tombol_start(self):
        global cameraIndex
            
        cameraIndex = self.boxKameraRegistrasi.currentIndex()
        self.btnStartRegistrasi.setEnabled(False)      
        self.btnPauseRegistrasi.setEnabled(True)    
        self.btnRegister.setEnabled(True)      
        self.btnStopRegistrasi.setEnabled(True)     
        self.thread.start()
            
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
        self.thread.stop()
        self.clear_label()
        self.btnStartRegistrasi.setEnabled(True)
        self.btnPauseRegistrasi.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(False)
        self.btnRegister.setEnabled(False)        
    
    def tombol_register(self):
        global aligned_img
        
        if self.lnLokasiSimpanDB.text() == "":
            QMessageBox.information(None, "Error", "Mohon masukkan folder penyimpanan database.")
        elif self.lnNamaWajah.text() == "":
            QMessageBox.information(None, "Error", "Mohon isi nama wajah yang akan diregistrasi.")
        elif self.btnNamaWajah.text() == "Terapkan":
            QMessageBox.information(None, "Error", "Klik tombol 'Terapkan' pada nama wajah terlebih dahulu.")
        else:             
            # Simpan gambar wajah ke folder database 
            duplikat = False
            now = datetime.datetime.now()
            time_now = now.strftime("%H%M%S")

            lokasi_pickle = self.lnLokasiSimpanDB.text()
            
            # Simpan database dalam format pickle
            database = {}

            if os.path.isfile(lokasi_pickle):
                pickle_database = open(lokasi_pickle, "rb")
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
                    lokasi_pickle = self.lnLokasiSimpanDB.text()
                    pickle_database = open(lokasi_pickle, "wb")
                    pickle.dump(database, pickle_database)
                    pickle_database.close()  
                    QMessageBox.information(None, "Info", 'Wajah "' + self.lnNamaWajah.text() + '" berhasil ditambahkan!')
            else:
                QMessageBox.information(None, "Error", "Gagal.")
            pickle_database = open(lokasi_pickle, "rb")
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

   
    def tombol_exit(self):
        sys.exit()

    def dialog_edit_database(self):
        file_database = QFileDialog.getOpenFileName(self, "Masukkan file database", "", "Pickle File (*.pkl)")
        
        if file_database[0] != "":
            self.lnEditFileDB.setText(str(file_database[0]))            
            
            lokasi_pickle = self.lnEditFileDB.text()
            pickle_database = open(lokasi_pickle, "rb")
            database = pickle.load(pickle_database)
            pickle_database.close()            
            if self.database_keys == [] and database != {}:
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
                QMessageBox.information(None, "Error", "File pickle tidak dapat dibaca. Buat ulang database melalui menu registrasi wajah.")   
        else:
            self.listDatabase.clear()
    def edit_nama_wajah(self):
        if self.btnEditNama.text() == "Terapkan":
            
            if self.lnEditFileDB.text() == "":
                QMessageBox.information(None, "Error", "Pilih file database terlebih dahulu!") 
            else:
                if self.lnEditNama.text() == "":
                    QMessageBox.information(None, "Error", "Nama wajah tidak boleh kosong!")
                else:
                    if self.database_keys[self.database_index].split("_")[1] == self.lnEditNama.text():
                        QMessageBox.information(None, "Error", "Nama yang dimasukkan sama dengan sebelumnya.")
                    elif "_" in self.lnEditNama.text():
                        QMessageBox.information(None, "Error", 'Tidak dapat menggunakan simbol "_"')
                    else:
                        self.btnPrevFrame.setEnabled(True)
                        self.btnNextFrame.setEnabled(True)
                        self.btnHapusFrame.setEnabled(True)
                        self.btnEditNama.setText("Ganti")
                        self.lnEditNama.setEnabled(False)
                        self.btnBatalEdit.setEnabled(False)
                        lokasi_pickle = self.lnEditFileDB.text()
                        pickle_database = open(lokasi_pickle, "rb")
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
                        pickle_database = open(lokasi_pickle, "wb")
                        pickle.dump(new_database_2, pickle_database)
                        pickle_database.close()
                        

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
        lokasi_pickle = self.lnEditFileDB.text()
        pickle_database = open(lokasi_pickle, "rb")
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
        lokasi_pickle = self.lnEditFileDB.text()
        pickle_database = open(lokasi_pickle, "rb")
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
        lokasi_pickle = self.lnEditFileDB.text()
        pickle_database = open(lokasi_pickle, "rb")
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
                self.similarFace.clear()
                self.similarFace.setText("Similar Face")
                QMessageBox.information(None, "Error", "Semua data wajah sudah dihapus. Silakan tambahkan data baru melalui menu registrasi wajah.")
        else:
            self.similarFace.clear()
            self.similarFace.setText("Similar Face")
            QMessageBox.information(None, "Error", "Data wajah kosong. Silakan tambahkan data baru melalui menu registrasi wajah.")

        pickle_database = open(lokasi_pickle, "wb")
        pickle.dump(database, pickle_database)
        pickle_database.close()
    
    def display_nama_wajah(self, nama):
        nama = nama.split("_")[1]
        self.lnEditNama.setText(nama)
        
    def clear_label(self):
        if self.pause:
            pass
        else:
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
        #self.align.adjustSize()
        self.face.setPixmap(qt_img)
        #self.align.setScaledContents(True)    
   
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
        #self.align.adjustSize()
        self.dbFace.setPixmap(qt_img)
        #self.align.setScaledContents(True)
    
def main():
    app = QApplication([])
    window = MyGUI()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()