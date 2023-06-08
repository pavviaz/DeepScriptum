import main_window_exp
import sys
from PIL import ImageGrab, Image

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QApplication,
    QFileDialog
)

import numpy as np
import cv2
from PIL import ImageGrab

from run_model import predict


ANIM_X_COORD = 960
ANIM_DURATION = 330


def make_fix_size(numpy_image, RESIZE_W=940, RESIZE_H=110):
    img = Image.fromarray(np.uint8(numpy_image)).convert('RGB')

    if img.size[0] <= RESIZE_W and img.size[1] <= RESIZE_H:
        background = Image.new('RGB', (RESIZE_W, RESIZE_H), (255, 255, 255))
        background.paste(img, (int(RESIZE_W / 2) - int(img.size[0] / 2), int(RESIZE_H / 2) - int(img.size[1] / 2)))
        return np.array(background)
    else:
        return None
    

def crop_pad_image(img):
    old_im = Image.fromarray(img)
    img_data = np.asarray(old_im, dtype=np.uint8)  # height, width
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:
        return

    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max + 1, y_max + 1))
    return old_im


# Refer to https://github.com/harupy/snipping-tool
class SnippingWidget(QMainWindow):
    is_snipping = False

    def __init__(self, parent=None, app=None):
        super(SnippingWidget, self).__init__()
        self.parent = parent
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        self.screen = app.primaryScreen()
        self.setGeometry(0, 0, self.screen.size().width(), self.screen.size().height())
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.onSnippingCompleted = None

    def start(self):
        SnippingWidget.is_snipping = True
        self.setWindowOpacity(0.3)
        QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.showFullScreen()

    def paintEvent(self, event):
        if SnippingWidget.is_snipping:
            brush_color = (128, 128, 255, 100)
            lw = 3
            opacity = 0.3
        else:
            self.begin = QtCore.QPoint()
            self.end = QtCore.QPoint()
            brush_color = (0, 0, 0, 0)
            lw = 0
            opacity = 0

        self.setWindowOpacity(opacity)
        qp = QtGui.QPainter(self)
        qp.setPen(QtGui.QPen(QtGui.QColor('black'), lw))
        qp.setBrush(QtGui.QColor(*brush_color))
        rect = QtCore.QRectF(self.begin, self.end)
        qp.drawRect(rect)

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = self.begin
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        SnippingWidget.is_snipping = False
        QApplication.restoreOverrideCursor()
        x1 = min(self.begin.x(), self.end.x())
        y1 = min(self.begin.y(), self.end.y())
        x2 = max(self.begin.x(), self.end.x())
        y2 = max(self.begin.y(), self.end.y())
        
        self.close()
        self.repaint()
        QApplication.processEvents()
        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))

        try:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except:
            img = None
        
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        
        if self.onSnippingCompleted is not None:
            self.onSnippingCompleted(img)


class OCR_Thread(QtCore.QThread):
    prediction_signal = QtCore.pyqtSignal(str)
    def  __init__(self, img, temp, decoder_type=None, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.img = img
        self.temp = temp
        
    def run(self):
        for pred in predict(self.img, self.temp):
            self.prediction_signal.emit(pred)


class MainWindow(QMainWindow, main_window_exp.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(QtCore.QSize(self.size().width() // 2, self.size().height()))
        
        self.first_page_objects = [el for el in vars(self) if "__f" in el]
        self.second_page_objects = [el for el in vars(self) if "__s" in el]
        
        self.snippingWidget = SnippingWidget(app=QApplication.instance())
        self.snippingWidget.onSnippingCompleted = self.__onSnippingCompleted
        self.screenshot_button__f.clicked.connect(self.snipArea)
        self.load_button__f.clicked.connect(self.load_image)
        self.temperature_slider__f.valueChanged.connect(self.temp_slider_value_changed)
        
        self.image_changed = False
        
        self.ocr_button__f.clicked.connect(self.start_ocr)
        self.back_button__s.clicked.connect(self.back_to_menu)
        
        # self.lastCommitOwner.clicked.connect(lambda: self.show_profile(self.lastCommitOwner.text().split(" ")[-1]))
    
    def __first_page_unload_anims(self):
        anims = []
        for el in self.first_page_objects:
            obj = getattr(self, el)
            anim = QtCore.QPropertyAnimation(obj, b'pos')
            pos = obj.pos()
            anim.setStartValue(pos)
            anim.setEndValue(QtCore.QPoint(pos.x() - ANIM_X_COORD, pos.y()))
            anim.setDuration(ANIM_DURATION)
            anims.append(anim)
        return anims

    def __first_page_load_anims(self):
        anims = []
        for el in self.first_page_objects:
            obj = getattr(self, el)
            anim = QtCore.QPropertyAnimation(obj, b'pos')
            pos = obj.pos()
            anim.setStartValue(pos)
            anim.setEndValue(QtCore.QPoint(pos.x() + ANIM_X_COORD, pos.y()))
            anim.setDuration(ANIM_DURATION)
            anims.append(anim)
        return anims
    
    def __second_page_load_anims(self):
        anims = []
        for el in self.second_page_objects:
            obj = getattr(self, el)
            anim = QtCore.QPropertyAnimation(obj, b'pos')
            pos = obj.pos()
            anim.setStartValue(pos)
            anim.setEndValue(QtCore.QPoint(pos.x() - ANIM_X_COORD, pos.y()))
            anim.setDuration(ANIM_DURATION)
            anims.append(anim)
        return anims
    
    def __second_page_unload_anims(self):
        anims = []
        for el in self.second_page_objects:
            obj = getattr(self, el)
            anim = QtCore.QPropertyAnimation(obj, b'pos')
            pos = obj.pos()
            anim.setStartValue(pos)
            anim.setEndValue(QtCore.QPoint(pos.x() + ANIM_X_COORD, pos.y()))
            anim.setDuration(ANIM_DURATION)
            anims.append(anim)
        return anims

    def start_ocr(self):
        if not self.image_changed:
            QMessageBox.warning(self, "Не выбрано изображение", "Выберите изображение с диска или сделайте снимок экрана")
            return
        self.f_page_unload_anims = self.__first_page_unload_anims()
        self.s_page_load_anims = self.__second_page_load_anims()
        
        [anim.start() for anim in self.f_page_unload_anims]
        [anim.start() for anim in self.s_page_load_anims]
                
        self.ocr = OCR_Thread(self.processed_image, self.temperature_slider__f.value() / 100)
        self.ocr.start()
        self.ocr.prediction_signal.connect(lambda x: self.latex_label__s.setText(x), QtCore.Qt.QueuedConnection)
        self.stop_ocr_button__s.clicked.connect(lambda _: self.ocr.terminate())
    
    def back_to_menu(self):
        self.ocr.terminate()
        self.f_page_load_anims = self.__first_page_load_anims()
        self.s_page_unload_anims = self.__second_page_unload_anims()
        
        [anim.start() for anim in self.s_page_unload_anims]
        [anim.start() for anim in self.f_page_load_anims]
    
    def __image_crop_pad_set(self, image_np):
        # image_np = np.array(Image.fromarray(image_np, mode="RGB").convert("L").convert("RGB"))
        cropped = crop_pad_image(image_np)
        if not cropped:
            QMessageBox.critical(self, "Ошибка снимка экрана", "На скриншоте отсутствует формула")
            return
        
        padded = make_fix_size(cropped)
        if padded is None:
            QMessageBox.critical(self, "Ошибка паддинга", "Формула слишком большая")
            return

        image = QtGui.QImage(padded, padded.shape[1], padded.shape[0], padded.strides[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.input_image__f.setPixmap(pixmap)
        self.input_image__s.setPixmap(pixmap)
        
        self.image_changed = True
        self.processed_image = padded
    
    def __onSnippingCompleted(self, frame):
        self.setWindowState(QtCore.Qt.WindowActive)
        if frame is None:
            return
        
        self.__image_crop_pad_set(frame)

    def snipArea(self):
        self.setWindowState(QtCore.Qt.WindowMinimized)
        self.snippingWidget.start()
        
    def load_image(self):
        fname = QFileDialog.getOpenFileName(
            self, 'Изображение', "C:/Users/shace/Documents/GitHub/im2latex/datasets/images_150/1.png", "Image files (*.jpg *.png)")
        if not fname[0]:
            return
        frame = cv2.imread(fname[0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        self.__image_crop_pad_set(frame)
    
    def temp_slider_value_changed(self):
        self.temp_slider_value__f.setText(f"{self.temperature_slider__f.value()} %")


if __name__ == "__main__":
    la2ex = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    window.show()   
    la2ex.exec_()