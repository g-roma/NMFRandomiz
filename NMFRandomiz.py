import sys, os, gc
import numpy as np
import matplotlib
from PySide import QtGui, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.colors import LinearSegmentedColormap
from ui import Ui_MainWindow
from untwist.transforms.stft import STFT, ISTFT
from untwist.data import Wave
from untwist.factorizations import NMF

class NMFRandomiz(QtGui.QMainWindow):
    def __init__(self):
        super(NMFRandomiz, self).__init__()
        self.playing = False
        self.init_ui()
        self.source_wave = None
        self.dest_wave =None
        self.K = 4
        self.spectrogram = None
        self.original_spectrogram = None
        self.stft = STFT()
        self.istft = ISTFT()
        self.nmf = NMF(self.K)
        self.shuffle_k = False
        self.shuffle_freq = False
        self.shuffle_time = False
        
    def init_ui(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.fig = Figure()        
        self.canvas = FigureCanvas(self.fig)
        self.ui.figure_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, 
            self.ui.navigation_widget)
        self.ui.navigation_layout.addWidget(self.toolbar)
        
        self.ui.play_btn.clicked.connect(self.play)
        self.ui.stop_btn.clicked.connect(self.stop)
        self.ui.load_btn.clicked.connect(self.load_audio_file)
        self.ui.save_btn.clicked.connect(self.save_audio_file)
        
        self.ui.num_comps_input_text.textChanged.connect(self.set_K)

        self.ui.randomizK_cb.clicked.connect(self.set_randK)   
        self.ui.randomizfreq_cb.clicked.connect(self.set_randF)   
        self.ui.randomiztime_cb.clicked.connect(self.set_randT)
        self.ui.computenmf_btn.clicked.connect(self.compute_NMF)
        
        self.axes = self.fig.add_axes([0,0,1,1])
        
    def set_randK(self):
        self.shuffle_k = self.ui.randomizK_cb.isChecked()
        
    def set_randF(self):
        self.shuffle_freq = self.ui.randomizfreq_cb.isChecked()
        
    def set_randT(self):
        self.shuffle_time = self.ui.randomiztime_cb.isChecked()

    def set_K(self):
        try:
            self.K = int(self.ui.num_comps_input_text.text())
            self.nmf = NMF(self.K)
            print self.K
        except:
            pass
                   
    def render_spectrogram(self):
        self.axes.clear()
        self.axes.imshow(
            20*np.log10(
                self.spectrogram.magnitude()
            ), 
            origin="low", aspect="auto", vmin = -60, cmap="Greys"
        )     
        self.canvas.draw()
        
    def load_audio_file(self):
        path, _ = QtGui.QFileDialog.getOpenFileName(self, 
            "Load an audio file", os.getcwd())
        w = Wave.read(path)
        if len(w.shape)>1:
            w = w[:,0]
        self.source_wave = w[:,np.newaxis]
        self.analyze()
        self.original_spectrogram = self.spectrogram

    def save_audio_file(self):
        path, _ = QtGui.QFileDialog.getSaveFileName(self, 
            "Save", os.getcwd())
        self.dest_wave = self.istft.process(self.spectrogram)
        self.dest_wave.write(path)
       
    def analyze(self):
        self.spectrogram = self.stft.process(self.source_wave)
        self.spectrogram[:,-1] = self.spectrogram[:,-2]
        self.render_spectrogram()

    def compute_NMF(self):
        [W,H, err] = self.nmf.process(self.original_spectrogram.magnitude())
        V = np.dot(W,H)
        k_index = np.arange(self.K)
        freq_index = np.arange(V.shape[0])
        time_index = np.arange(H.shape[1])
        if self.shuffle_k:
            np.random.shuffle(k_index)
            W = W[:,k_index]
        if self.shuffle_freq:
            np.random.shuffle(freq_index)
            W = W[freq_index,:]
        if self.shuffle_time:
            np.random.shuffle(time_index)
            H = H[:,time_index]
        Vr = np.dot(W,H)
        R = np.nan_to_num(Vr/V)
        self.spectrogram = self.original_spectrogram * R
        self.render_spectrogram()

    def play(self):
        if self.playing: self.stop()
        self.dest_wave = self.istft.process(self.spectrogram)
        self.dest_wave.play()
        self.playing = True
           
    def stop(self): 
        if self.playing:
            self.dest_wave.stop()
            self.playing = False

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    main = NMFRandomiz()    
    main.show()
    sys.exit(app.exec_())    