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
import math
from PIL import Image

class VideoThread(QThread):    
    face_signal = pyqtSignal(np.ndarray)    
    db_face_signal = pyqtSignal(np.ndarray)

    global cameraIndex
    
    def __init__(self, my_gui):
        super().__init__()
        self.my_gui = my_gui

    def run(self):     
        global aligned_img   
        if cameraIndex == 0:
            cap = cv2.VideoCapture(cameraIndex)
        else:
            cap = cv2.VideoCapture(cameraIndex,cv2.CAP_DSHOW)
        scale_percent = 100

        frame_w = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale_percent / 100)
        frame_h = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale_percent / 100)   
        dim = (frame_w, frame_h)

        self.isActive = True
        tm = cv2.TickMeter()
        
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
                face_img = self.find_face(original_img, detected_img)
                landmarks = self.find_landmarks(detected_img)
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
    
    def find_face(self, input, faces, thickness=2):      
        face_img = input.copy()   
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            for i in range(len(coords)):
                if coords[i] < 0:
                    coords[i] = 0            
            face_img = face_img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]                
            if len(face_img) != 0:                
                return face_img   
                
    def find_landmarks(self, face):
        for det in face[1]:
            landmarks = det[4:14].astype(np.int32).reshape((5,2))
            if len(landmarks) != 0:
                return landmarks
    
    def euclidean_distance(self, a, b):
        x1 = a[0]; y1 = a[1]
        x2 = b[0]; y2 = b[1]
        return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))     

    def align_face(self, face_img, landmarks):
        if landmarks is not None and len(face_img) != 0:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            if left_eye[1] < right_eye[1]:
                third_point = (right_eye[0], left_eye[1])
                direction = -1
            else:
                third_point = (left_eye[0], right_eye[1])
                direction = 1

            a = self.euclidean_distance(left_eye, third_point)
            b = self.euclidean_distance(right_eye, left_eye)
            c = self.euclidean_distance(right_eye, third_point)
            cos_a = (b*b + c*c - a*a) / (2*b*c)
            angle = (np.arccos(cos_a) * 180) / math.pi
            if direction == -1:
                angle = 90 - angle
            direction = -1 * direction
            # ERROR!!!!!!!
            new_img = Image.fromarray(face_img)
            return np.array(new_img.rotate(direction * angle))
        else:
            return None

    def stop(self):      
        #time.sleep(1)      
        self.quit()
        self.isActive = False
        self.my_gui.clear_label()       
    
            
class MyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("register.ui", self)
        self.show()

        self.file_model_deteksi = "face_detection_yunet.onnx"
        self.file_model_pengenalan = "face_recognition_sface.onnx"
        
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
        self.update_list_database()
        self.lnNamaWajah.clear()
        self.lnNamaWajah.setEnabled(True)
        self.btnNamaWajah.setText("Terapkan")
        
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
    
    def tombol_start(self):
        global cameraIndex
            
        cameraIndex = self.boxKameraRegistrasi.currentIndex()
        self.btnStartRegistrasi.setEnabled(False)      
        self.btnPauseRegistrasi.setEnabled(True)    
        self.btnRegister.setEnabled(True)      
        self.btnStopRegistrasi.setEnabled(True)   
        self.btnRegistrasi.setEnabled(False)
        self.btnEditDB.setEnabled(False)
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
                    QMessageBox.information(None, "Info", 'Wajah "' + self.lnNamaWajah.text() + '" berhasil ditambahkan!')
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
