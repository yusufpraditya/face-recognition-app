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