from shutil import ExecError
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt6 import uic, QtGui
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtMultimedia import *
from PyQt6.QtMultimediaWidgets import *
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
    detection_signal = pyqtSignal(np.ndarray)
    crop_signal = pyqtSignal(np.ndarray)
    alignment_signal = pyqtSignal(np.ndarray)
    original_face_signal = pyqtSignal(np.ndarray)
    similar_face_signal = pyqtSignal(np.ndarray)

    global cameraIndex, file_model_deteksi, file_model_pengenalan, lokasi_pickle
    
    def __init__(self, my_gui):
        super().__init__()
        self.my_gui = my_gui

    def run(self):     
        global aligned_img   
        if cameraIndex == 0:
            cap = cv2.VideoCapture(cameraIndex)
        if cameraIndex == 1:
            cap = cv2.VideoCapture(cameraIndex,cv2.CAP_DSHOW)
        if self.my_gui.mode_videofoto_pengenalan:
           cap = cv2.VideoCapture(self.my_gui.path_video)
           self.my_gui.panjang_frame_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.isActive = True
        tm = cv2.TickMeter()
        while self.isActive:
            count = 0
            tm.start()            
            if self.my_gui.pause:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.my_gui.frame_video)
                self.my_gui.pause = False
            _, original_img = cap.read()        
            self.my_gui.frame_video = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if self.my_gui.mode_pengenalan:
                pickle_database = open(lokasi_pickle, "rb")
                database = pickle.load(pickle_database)
                pickle_database.close()  

            model_yunet = YuNet(model_path=file_model_deteksi)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            model_yunet.set_input_size([w, h])

            model_sface = SFace(model_path=file_model_pengenalan)      
           
            #try:
            detected_img, face_img, landmarks = model_yunet.detect(original_img)
            aligned_img = model_yunet.align_face(face_img, landmarks)       
            face_feature = model_sface.feature(aligned_img)            
            if detected_img is not None and face_img is not None and landmarks is not None and aligned_img is not None and face_feature is not None:
                if self.my_gui.mode_pengenalan == False:
                    self.detection_signal.emit(detected_img)
                    self.crop_signal.emit(face_img)
                    self.alignment_signal.emit(aligned_img)
                else:
                    max_cosine = 0
                    cosine_similarity_threshold = float(self.my_gui.valSimilarity.text())
                    identity = 'unknown'
                    for key in database.keys():
                        name = key.split("_")[0]
                        if name != "img":
                            cosine_score = model_sface.match(face_feature, database[key])
                            if cosine_score > max_cosine:
                                max_cosine = cosine_score
                                identity = key
                    
                    str_max_cosine = "{:.3f}".format(round(max_cosine, 3))
                    self.my_gui.lcdSimilarity.display(str_max_cosine)
                    if max_cosine >= cosine_similarity_threshold:
                        identity_image = database["img_" + identity]
                        self.similar_face_signal.emit(identity_image)
                        identity = identity.split("_")[0]                        
                    else:
                        identity = 'unknown'
                    
                    self.my_gui.lineHasilPengenalan.setText(identity)
                    
                    self.detection_signal.emit(detected_img)
                    self.crop_signal.emit(face_img)
                    self.alignment_signal.emit(aligned_img)
                    self.original_face_signal.emit(aligned_img)
                    
            else:
                self.detection_signal.emit(original_img)
            tm.stop()
            fps = "{:.2f}".format(round(tm.getFPS(), 2))
            self.my_gui.lcdFPS.display(fps)
            
            #except Exception as e:
            #    print(e)            
            #    print("test")

    def stop(self):      
        #time.sleep(1)      
        self.quit()
        self.isActive = False
        self.my_gui.lcdSimilarity.display("0")
        self.my_gui.detection.clear()
        self.my_gui.detection.setText("Detection")
        self.my_gui.crop.clear()
        self.my_gui.crop.setText("Crop")
        self.my_gui.align.clear()
        self.my_gui.align.setText("Align")
        self.my_gui.originalFace.clear()
        self.my_gui.originalFace.setText("Original Face")
        self.my_gui.similarFace.clear()
        self.my_gui.similarFace.setText("Similar Face")
        self.my_gui.btnKameraPengenalan.setEnabled(True)
        self.my_gui.btnVideoFotoPengenalan.setEnabled(True)
    
            
class MyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("desain_v3.ui", self)
        self.show()
        
        self.pause = False
        self.stop = False
        self.mode_pengenalan = False
        self.mode_kamera_pengenalan = False
        self.mode_videofoto_pengenalan = False
        self.path_video = ""
        self.database_keys = []
        self.database_index = 0
        self.panjang_frame_video = 0
        self.frame_video = 0

        self.display_width = 100
        self.display_height = 100

        self.crop.setMaximumSize(self.crop.width(), self.crop.height())
        self.align.setMaximumSize(self.align.width(), self.align.height())
        
        self.pilihanTab.setCurrentIndex(0)

        self.pilihanTab.setEnabled(False)
        self.btnPauseRegistrasi.setEnabled(False)
        self.btnRegister.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(False)        

        # Pilih tab registrasi/pengenalan/edit database
        self.btnRegistrasi.clicked.connect(self.tab_registrasi)
        self.btnPengenalan.clicked.connect(self.tab_pengenalan)
        self.btnEditDB.clicked.connect(self.tab_edit_database)

        # Dialog file model deteksi wajah
        self.btnModelDeteksi.clicked.connect(self.dialog_deteksi_wajah)

        # Dialog file model pengenalan wajah
        self.btnModelPengenalan.clicked.connect(self.dialog_pengenalan_wajah)    

        # Ubah nilai threshold similarity
        self.btnSimilarity.clicked.connect(self.threshold_similarity)    

        # Pilih input (tab registrasi)
        self.btnKameraRegistrasi.clicked.connect(self.kamera_registrasi)
        self.btnFotoRegistrasi.clicked.connect(self.foto_registrasi)
        self.btnLokasiFoto.clicked.connect(self.lokasi_foto)    

        # Lokasi penyimpanan gambar wajah
        self.btnSimpanWajah.clicked.connect(self.dialog_simpan_database)

        # Edit nama wajah
        self.btnNamaWajah.clicked.connect(self.nama_wajah)
        self.btnEditNama.clicked.connect(self.edit_nama_wajah)
        self.btnBatalEdit.clicked.connect(self.batal_edit_nama)

        # Pilih input (tab pengenalan)
        self.btnKameraPengenalan.clicked.connect(self.kamera_pengenalan)
        self.btnVideoFotoPengenalan.clicked.connect(self.video_foto_pengenalan)
        self.btnLokasiVideoFoto.clicked.connect(self.lokasi_video_foto_pengenalan)

        # Lokasi database
        self.btnLokasiDB.clicked.connect(self.dialog_lokasi_database)

        # Tombol-tombol Tab Registrasi
        self.btnStartRegistrasi.clicked.connect(self.tombol_start)
        self.btnPauseRegistrasi.clicked.connect(self.tombol_pause)
        self.btnStopRegistrasi.clicked.connect(self.tombol_stop)
        self.btnRegister.clicked.connect(self.tombol_register)    

        # Tombol-tombol Tab Pengenalan
        self.btnStartPengenalan.clicked.connect(self.tombol_start_pengenalan)
        self.btnPausePengenalan.clicked.connect(self.tombol_pause_pengenalan)
        self.btnStopPengenalan.clicked.connect(self.tombol_stop_pengenalan)

        # Lokasi file database yang akan diedit
        self.btnEditFileDB.clicked.connect(self.dialog_edit_database)

        # Tombol-tombol Tab Edit Database
        self.btnNextFrame.clicked.connect(self.tombol_next_frame)
        self.btnPrevFrame.clicked.connect(self.tombol_prev_frame)
        self.btnHapusFrame.clicked.connect(self.tombol_hapus_frame)

        # Tombol keluar
        self.btnExit.clicked.connect(self.tombol_exit)
        

        self.thread = VideoThread(self)
        self.thread.detection_signal.connect(self.update_detection)
        self.thread.crop_signal.connect(self.update_crop)
        self.thread.alignment_signal.connect(self.update_align)
        self.thread.original_face_signal.connect(self.update_original)
        self.thread.similar_face_signal.connect(self.update_similar)

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
    
    def threshold_similarity(self):
        if self.btnSimilarity.text() == "Terapkan":
            self.btnSimilarity.setText("Ganti")
            self.valSimilarity.setEnabled(False)
        else:
            self.btnSimilarity.setText("Terapkan")
            self.valSimilarity.setEnabled(True)

    def tab_registrasi(self):
        self.mode_pengenalan = False
        self.pilihanTab.setEnabled(True)
        self.pilihanTab.setCurrentIndex(0)
        
        self.btnStartRegistrasi.setEnabled(False)
        self.btnPauseRegistrasi.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(False)

        self.btnSimilarity.setEnabled(False)
        self.valSimilarity.setEnabled(False)
    
    def tab_pengenalan(self):
        self.mode_pengenalan = True
        self.pilihanTab.setEnabled(True)
        self.pilihanTab.setCurrentIndex(1)

        self.btnStartPengenalan.setEnabled(False)
        self.btnPausePengenalan.setEnabled(False)
        self.btnStopPengenalan.setEnabled(False)

        self.btnSimilarity.setEnabled(True)
        self.valSimilarity.setEnabled(True)
    
    def tab_edit_database(self):
        self.mode_pengenalan = False
        self.pilihanTab.setEnabled(True)
        self.pilihanTab.setCurrentIndex(2)

        self.btnPrevFrame.setEnabled(False)
        self.btnNextFrame.setEnabled(False)
        self.btnHapusFrame.setEnabled(False)

        self.btnSimilarity.setEnabled(False)
        self.valSimilarity.setEnabled(False)
    
    def kamera_registrasi(self):
        self.boxKameraRegistrasi.setEnabled(True)
        self.lnFotoRegistrasi.setEnabled(False)
        self.btnLokasiFoto.setEnabled(False)
        self.boxKameraRegistrasi.clear()
        self.lnFotoRegistrasi.clear()

        self.btnStartRegistrasi.setEnabled(True)

        # Tambah list kamera ke combobox
        cameraList = QMediaDevices.videoInputs()        
        for c in cameraList:
            self.boxKameraRegistrasi.addItem(c.description())
    
    def foto_registrasi(self):
        self.boxKameraRegistrasi.setEnabled(False)
        self.lnFotoRegistrasi.setEnabled(True)
        self.btnLokasiFoto.setEnabled(True)
        self.boxKameraRegistrasi.clear()

        self.btnStartRegistrasi.setEnabled(False)
        self.btnPauseRegistrasi.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(False)

    def lokasi_foto(self):
        img_file = QFileDialog.getOpenFileName(self, "Masukkan gambar subjek yang akan diregistrasi", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if img_file:
            gambar_subjek = str(img_file[0])
            self.lnFotoRegistrasi.setText(gambar_subjek)
            self.process_image(gambar_subjek)
    
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
        if self.lnDeteksi.text() == "" or self.lnPengenalan.text() == "":
            QMessageBox.information(None, "Error", "Mohon masukkan file model deteksi & pengenalan pada bagian Setting.")
        else:
            
            cameraIndex = self.boxKameraRegistrasi.currentIndex()
            self.btnStartRegistrasi.setEnabled(False)      
            self.btnPauseRegistrasi.setEnabled(True)    
            self.btnRegister.setEnabled(True)      
            self.btnStopRegistrasi.setEnabled(True)
            self.btnKameraRegistrasi.setEnabled(False)
            self.btnFotoRegistrasi.setEnabled(False)
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
        self.detection.clear()
        self.detection.setText("Detection")
        self.crop.clear()
        self.crop.setText("Crop")
        self.align.clear()
        self.align.setText("Align")
        self.btnKameraRegistrasi.setEnabled(True)
        self.btnFotoRegistrasi.setEnabled(True)
        self.btnStartRegistrasi.setEnabled(True)
        self.btnPauseRegistrasi.setEnabled(False)
        self.btnStopRegistrasi.setEnabled(False)
        self.btnRegister.setEnabled(False)        
    
    def tombol_register(self):
        global aligned_img, file_model_pengenalan
        
        if self.lnLokasiSimpanDB.text() == "":
            QMessageBox.information(None, "Error", "Mohon masukkan folder penyimpanan database.")
        elif self.lnNamaWajah.text() == "":
            QMessageBox.information(None, "Error", "Mohon isi nama wajah yang akan diregistrasi.")
        elif self.btnNamaWajah.text() == "Terapkan":
            QMessageBox.information(None, "Error", "Klik tombol 'Terapkan' pada nama wajah terlebih dahulu.")
        else:             
            # Simpan gambar wajah ke folder database  
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
            model_pengenalan = cv2.FaceRecognizerSF.create(file_model_pengenalan, "")    
            fitur_wajah = model_pengenalan.feature(aligned_img)
            database[nama_wajah] = fitur_wajah

            lokasi_pickle = self.lnLokasiSimpanDB.text()
            pickle_database = open(lokasi_pickle, "wb")
            pickle.dump(database, pickle_database)
            pickle_database.close()  
            
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
            QMessageBox.information(None, "Info", 'Wajah "' + self.lnNamaWajah.text() + '" berhasil ditambahkan!')               

    def kamera_pengenalan(self):          
        self.mode_kamera_pengenalan = True
        self.mode_videofoto_pengenalan = False
        self.boxKameraPengenalan.setEnabled(True)
        self.lnVideoFotoPengenalan.setEnabled(False)
        self.btnLokasiVideoFoto.setEnabled(False)
        self.boxKameraPengenalan.clear()
        self.lnVideoFotoPengenalan.clear()

        self.btnStartPengenalan.setEnabled(True)        

        # Tambah list kamera ke combobox
        cameraList = QMediaDevices.videoInputs()        
        for c in cameraList:
            self.boxKameraPengenalan.addItem(c.description())
    
    def video_foto_pengenalan(self):   
        self.mode_kamera_pengenalan = False
        self.mode_videofoto_pengenalan = True
        self.boxKameraPengenalan.setEnabled(False)
        self.lnVideoFotoPengenalan.setEnabled(True)
        self.btnLokasiVideoFoto.setEnabled(True)
        self.boxKameraPengenalan.clear()

        #self.btnStartPengenalan.setEnabled(True)
        self.btnPausePengenalan.setEnabled(False)
        self.btnStopPengenalan.setEnabled(False)

    def lokasi_video_foto_pengenalan(self):        
        img_videofoto_file = QFileDialog.getOpenFileName(self, "Masukkan video/foto yang akan dikenali", "", "Image/Video Files (*.jpg *.jpeg *.png *.bmp *.mp4)")
        if img_videofoto_file:
            path_file = str(img_videofoto_file[0])            
            self.lnVideoFotoPengenalan.setText(path_file)
            file_format = path_file.split('.')[-1]
            if file_format.lower() in ['jpg', 'jpeg', 'png', 'bmp']:
                self.btnStartPengenalan.setEnabled(False)           
                self.process_image(path_file)
            else:
                self.btnStartPengenalan.setEnabled(True)
                self.path_video = path_file
                #self.process_video(path_file)

    def dialog_lokasi_database(self):
        global lokasi_pickle
        
        file_database = QFileDialog.getOpenFileName(self, "Masukkan file database", "", "Pickle File (*.pkl)")
        
        if file_database[0] != "":
            self.lnLokasiDB.setText(file_database[0])
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

    def tombol_start_pengenalan(self):
        global cameraIndex        
        if self.lnDeteksi.text() == "" or self.lnPengenalan.text() == "":            
            QMessageBox.information(None, "Error", "Mohon masukkan file model deteksi & pengenalan pada bagian Setting.")        
        elif self.lnLokasiDB.text() == "":            
            QMessageBox.information(None, "Error", "Pilih lokasi folder database terlebih dahulu.")        
        else:            
            if self.lnVideoFotoPengenalan.text() == "":
                if self.mode_videofoto_pengenalan:
                    QMessageBox.information(None, "Error", "Pilih file video/foto yang akan dikenali terlebih dahulu.")
            
            if self.btnSimilarity.text() == "Terapkan":
                QMessageBox.information(None, "Error", "Klik tombol 'Terapkan' pada nilai threshold cosine similarity terlebih dahulu.")
            else:
                cameraIndex = self.boxKameraPengenalan.currentIndex()
                self.btnStartPengenalan.setEnabled(False)
                self.btnPausePengenalan.setEnabled(True)            
                self.btnStopPengenalan.setEnabled(True)
                self.btnKameraPengenalan.setEnabled(False)
                self.btnVideoFotoPengenalan.setEnabled(False)
                self.thread.start()

    def tombol_pause_pengenalan(self):
        self.pause = True
        self.stop = False
        self.btnStartPengenalan.setEnabled(True)
        self.btnPausePengenalan.setEnabled(False)
        self.btnStopPengenalan.setEnabled(True)
        self.thread.stop()

    def tombol_stop_pengenalan(self):
        self.pause = False
        self.stop = True
        self.btnStartPengenalan.setEnabled(True)
        self.btnPausePengenalan.setEnabled(False)
        self.btnStopPengenalan.setEnabled(False)
        self.thread.stop()              

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
                self.update_similar(database[self.database_keys[0]])
                self.display_nama_wajah(self.database_keys[self.database_index])  
            else:
                QMessageBox.information(None, "Error", "File pickle tidak dapat dibaca. Buat ulang database melalui menu registrasi wajah.")   
    
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
            self.update_similar(database[self.database_keys[self.database_index]])
            print(self.database_keys[self.database_index])
            self.display_nama_wajah(self.database_keys[self.database_index])  
        else:
            self.database_index = 0
            self.update_similar(database[self.database_keys[self.database_index]])
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
            self.update_similar(database[self.database_keys[self.database_index]])
            print(self.database_keys[self.database_index])
            self.display_nama_wajah(self.database_keys[self.database_index])  
        else:
            self.database_index = len(self.database_keys) - 1
            self.update_similar(database[self.database_keys[self.database_index]])
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
                    self.update_similar(database[self.database_keys[self.database_index]])
                    self.display_nama_wajah(self.database_keys[self.database_index])  
                else:
                    self.database_index -= 1
                    self.update_similar(database[self.database_keys[self.database_index]])
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
    
    def process_image(self, path_gambar):        
        global file_model_deteksi
        
        original_img = cv2.imread(path_gambar)

        model = YuNet(model_path=file_model_deteksi)
        h, w, _ = original_img.shape
        model.set_input_size([w, h])            
             
        try:     
            detected_img, face_img, landmarks = model.detect(original_img)   
            aligned_img = model.align_face(face_img, landmarks)
            if detected_img is not None and face_img is not None and landmarks is not None:    
                self.update_detection(detected_img)        
                self.update_crop(face_img)
                self.update_align(aligned_img)
            else:
                self.update_detection(original_img)  
        except Exception as e:
            print(e)                  

    @pyqtSlot(np.ndarray)
    def update_detection(self, cv_img): 
        h, w, _ = cv_img.shape

        # Resize
        if h > self.detection.height() or w > self.detection.width():
           h_ratio = self.detection.height() / h
           w_ratio = self.detection.width() / w
           scale_factor = min(h_ratio, w_ratio)
           h = int(h * scale_factor)
           w = int(w * scale_factor)
           dim = (w, h)
           cv_img = cv2.resize(cv_img, dim)

        bytes_per_line = 3 * w
        qt_format = QtGui.QImage(cv_img, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)        
        qt_img = QPixmap.fromImage(qt_format)        
        #self.detection.adjustSize()
        self.detection.setPixmap(qt_img)
        #self.detection.setScaledContents(True)

    @pyqtSlot(np.ndarray)
    def update_crop(self, face_img):
        face_img = face_img.copy()
        h, w, _ = face_img.shape

        # Resize
        if h > self.crop.height() or w > self.crop.width():
           h_ratio = self.crop.height() / h
           w_ratio = self.crop.width() / w
           scale_factor = min(h_ratio, w_ratio)
           h = int(h * scale_factor)
           w = int(w * scale_factor)
           dim = (w, h)
           face_img = cv2.resize(face_img, dim)

        bytes_per_line = 3 * w
        qt_format = QtGui.QImage(face_img, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)        
        qt_img = QPixmap.fromImage(qt_format)
        #self.crop.adjustSize()
        self.crop.setPixmap(qt_img)
        #self.crop.setScaledContents(True)

    @pyqtSlot(np.ndarray)
    def update_align(self, face_img):
        h, w, _ = face_img.shape

        # Resize
        if h > self.align.height() or w > self.align.width():
           h_ratio = self.align.height() / h
           w_ratio = self.align.width() / w
           scale_factor = min(h_ratio, w_ratio)
           h = int(h * scale_factor)
           w = int(w * scale_factor)
           dim = (w, h)
           face_img = cv2.resize(face_img, dim)

        bytes_per_line = 3 * w
        qt_format = QtGui.QImage(face_img, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        qt_img = QPixmap.fromImage(qt_format)
        #self.align.adjustSize()
        self.align.setPixmap(qt_img)
        #self.align.setScaledContents(True)
    
    @pyqtSlot(np.ndarray)
    def update_original(self, face_img):
        h, w, _ = face_img.shape

        # Resize
        if h > self.originalFace.height() or w > self.originalFace.width():
           h_ratio = self.originalFace.height() / h
           w_ratio = self.originalFace.width() / w
           scale_factor = min(h_ratio, w_ratio)
           h = int(h * scale_factor)
           w = int(w * scale_factor)
           dim = (w, h)
           face_img = cv2.resize(face_img, dim)

        bytes_per_line = 3 * w
        qt_format = QtGui.QImage(face_img, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        qt_img = QPixmap.fromImage(qt_format)
        #self.align.adjustSize()
        self.originalFace.setPixmap(qt_img)
        #self.align.setScaledContents(True)
    
    @pyqtSlot(np.ndarray)
    def update_similar(self, face_img):
        h, w, _ = face_img.shape

        # Resize
        if h > self.similarFace.height() or w > self.similarFace.width():
           h_ratio = self.similarFace.height() / h
           w_ratio = self.similarFace.width() / w
           scale_factor = min(h_ratio, w_ratio)
           h = int(h * scale_factor)
           w = int(w * scale_factor)
           dim = (w, h)
           face_img = cv2.resize(face_img, dim)

        bytes_per_line = 3 * w
        qt_format = QtGui.QImage(face_img, w, h, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        qt_img = QPixmap.fromImage(qt_format)
        #self.align.adjustSize()
        self.similarFace.setPixmap(qt_img)
        #self.align.setScaledContents(True)
    
def main():
    app = QApplication([])
    window = MyGUI()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()