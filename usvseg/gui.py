import sys
import os
import yaml
import numpy as np
import glob

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QDesktopWidget, QHBoxLayout, 
                             QVBoxLayout, QFormLayout, QLabel, QLineEdit,
                             QPushButton, QFileDialog, QApplication, QMessageBox)
from PyQt5.QtGui import QDoubleValidator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.patches

from . import func as usvseg_func
#import func as usvseg_func

script_dir = os.path.dirname(__file__)
prm_file_path = script_dir + '/usvseg_prm.yaml'

def defalut_param():
    
    P = {'fftsize': int(512), 'timestep': float(0.0005), 'freqmin': float(30000), 'freqmax': float(120000),
         'threshval': float(4.5), 'durmin': float(0.005), 'durmax': float(0.3), 'gapmin': float(0.03),
         'margin': float(0.015), 'wavfileoutput': True, 'imageoutput': True, 'imagetype': int(0),
         'traceoutput': False, 'readsize': float(10.0), 'mapL': float(1000),
         'mapH': float(6000), 'synthsndoutput': False, 'bandwidth':int(9), 'liftercutoff':int(3)}
    
    return P

class ProcWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(usvseg_func.FigData)
    
    def __init__(self, flag_multi, params, fp, savefp=None, outp=None, fname_audiblewav=None):

        super().__init__()

        self.flag_multi = flag_multi
        self.params = params
        self.fp = fp
        self.savefp = savefp
        self.outp = outp
        self.fname_audiblewav = fname_audiblewav
        self.i_file = -1

    def run(self):

        self.running = True
        
        if not self.flag_multi:
            usvseg_func.proc_wavfile(self.params, self.fp, self.savefp, self.outp, self.fname_audiblewav, ui_thread=self)
        else:
            for i_file, fp in enumerate(self.fp):
                self.i_file = i_file
                savefp = os.path.splitext(fp)[0] + '_dat.csv'
                outp = os.path.splitext(fp)[0]
                if self.params['synthsndoutput']:
                    fname_audiblewav = os.path.splitext(fp)[0] + '_syn.wav'
                else:
                    fname_audiblewav = None
                usvseg_func.proc_wavfile(self.params, fp, savefp, outp, fname_audiblewav, ui_thread=self)

                if self.thread().isInterruptionRequested():
                    break


        self.finished.emit()
        
        self.running = False

class MainWindow(QWidget):

    def __init__(self):

        super().__init__()

        self.worker = None

        if not os.path.exists(prm_file_path):
            self.params = defalut_param()
            with open(prm_file_path, 'w') as f:
                yaml.dump(self.params, f)
        else:
            with open(prm_file_path, 'r') as f:
                self.params = yaml.load(f, Loader=yaml.SafeLoader)
        
        self.resize(1000, 750)
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.setMinimumSize(800, 600)

        self.layout_main = QHBoxLayout()
        self.layout_setting = QVBoxLayout()
        self.layout_fig = QVBoxLayout()
        self.layout_form = QFormLayout()

        self.setting_widget = QWidget()
        self.setting_widget.setLayout(self.layout_setting)
        self.setting_widget.setFixedWidth(160)

        self.layout_main.addWidget(self.setting_widget)
        self.layout_setting.addLayout(self.layout_form)

        setting_ui_width = 60
        setting_ui_height = 30

        self.label_fs = QLabel('-')
        self.label_fs.setAlignment(Qt.AlignCenter)
        self.label_fs.setFixedWidth(setting_ui_width)
        self.label_fs.setFixedHeight(setting_ui_height)
        self.layout_form.addRow(QLabel('sampling\n(kHz)'), self.label_fs)

        self.edit_timestep = QLineEdit(str(self.params['timestep']*1000))
        self.edit_timestep.setValidator(QDoubleValidator(0.0,  100.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_timestep.setAlignment(Qt.AlignRight)
        self.edit_timestep.setFixedWidth(setting_ui_width)
        self.edit_timestep.setFixedHeight(setting_ui_height)
        self.layout_form.addRow(QLabel('time step\n(ms)'), self.edit_timestep)

        self.edit_freqmin = QLineEdit(str(self.params['freqmin']/1000))
        self.edit_freqmin.setValidator(QDoubleValidator(0.0,  1000.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_freqmin.setAlignment(Qt.AlignRight)
        self.edit_freqmin.setFixedWidth(setting_ui_width)
        self.edit_freqmin.setFixedHeight(setting_ui_height)
        self.layout_form.addRow(QLabel('freq min\n(kHz)'), self.edit_freqmin)

        self.edit_freqmax = QLineEdit(str(self.params['freqmax']/1000))
        self.edit_freqmax.setValidator(QDoubleValidator(0.0,  1000.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_freqmax.setAlignment(Qt.AlignRight)
        self.edit_freqmax.setFixedWidth(setting_ui_width)
        self.edit_freqmax.setFixedHeight(setting_ui_height)
        self.layout_form.addRow(QLabel('freq max\n(kHz)'), self.edit_freqmax)

        self.edit_thresh = QLineEdit(str(self.params['threshval']))
        self.edit_thresh.setValidator(QDoubleValidator(0.0,  100.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_thresh.setAlignment(Qt.AlignRight)
        self.edit_thresh.setFixedWidth(setting_ui_width)
        self.edit_thresh.setFixedHeight(setting_ui_height)
        self.layout_form.addRow(QLabel('threshold\n(SD)'), self.edit_thresh)

        self.edit_durmin = QLineEdit(str(self.params['durmin']*1000))
        self.edit_durmin.setValidator(QDoubleValidator(0.0,  1000.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_durmin.setAlignment(Qt.AlignRight)
        self.edit_durmin.setFixedWidth(setting_ui_width)
        self.edit_durmin.setFixedHeight(setting_ui_height)
        self.layout_form.addRow(QLabel('dur min\n(ms)'), self.edit_durmin)

        self.edit_durmax = QLineEdit(str(self.params['durmax']*1000))
        self.edit_durmax.setValidator(QDoubleValidator(0.0,  1000.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_durmax.setAlignment(Qt.AlignRight)
        self.edit_durmax.setFixedWidth(setting_ui_width)
        self.edit_durmax.setFixedHeight(setting_ui_height)
        self.layout_form.addRow(QLabel('dur max\n(ms)'), self.edit_durmax)

        self.edit_gapmin = QLineEdit(str(self.params['gapmin']*1000))
        self.edit_gapmin.setValidator(QDoubleValidator(0.0,  1000.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_gapmin.setAlignment(Qt.AlignRight)
        self.edit_gapmin.setFixedWidth(setting_ui_width)
        self.edit_gapmin.setFixedHeight(setting_ui_height)
        self.layout_form.addRow(QLabel('gap min\n(ms)'), self.edit_gapmin)

        self.edit_margin = QLineEdit(str(self.params['margin']*1000))
        self.edit_margin.setValidator(QDoubleValidator(0.0,  1000.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_margin.setAlignment(Qt.AlignRight)
        self.edit_margin.setFixedWidth(setting_ui_width)
        self.edit_margin.setFixedHeight(setting_ui_height)
        self.layout_form.addRow(QLabel('margin\n(ms)'), self.edit_margin)

        self.btn_wavfileoutput = QPushButton('on' if self.params['wavfileoutput'] else 'off')
        self.btn_wavfileoutput.setCheckable(True)
        self.btn_wavfileoutput.setChecked(self.params['wavfileoutput'])
        self.btn_wavfileoutput.setFixedWidth(setting_ui_width)
        self.btn_wavfileoutput.setFixedHeight(setting_ui_height)
        self.btn_wavfileoutput.clicked.connect(lambda: self.update_toggle_buttons())
        self.layout_form.addRow(QLabel('wavfile\noutput'), self.btn_wavfileoutput)
        
        self.btn_imageoutput = QPushButton('on' if self.params['imageoutput'] else 'off')
        self.btn_imageoutput.setCheckable(True)
        self.btn_imageoutput.setChecked(self.params['imageoutput'])
        self.btn_imageoutput.setFixedWidth(setting_ui_width)
        self.btn_imageoutput.setFixedHeight(setting_ui_height)
        self.btn_imageoutput.clicked.connect(lambda: self.update_toggle_buttons())
        self.layout_form.addRow(QLabel('image\noutput'), self.btn_imageoutput)

        self.btn_imagetype = QPushButton('orig' if self.params['imagetype']==0 else 'flat')
        self.btn_imagetype.setCheckable(True)
        self.btn_imagetype.setChecked(self.params['imagetype']==1)
        self.btn_imagetype.setFixedWidth(setting_ui_width)
        self.btn_imagetype.setFixedHeight(setting_ui_height)
        self.btn_imagetype.clicked.connect(lambda: self.update_toggle_buttons())
        self.layout_form.addRow(QLabel('image\ntype'), self.btn_imagetype)

        self.btn_traceoutput = QPushButton('on' if self.params['traceoutput'] else 'off')
        self.btn_traceoutput.setCheckable(True)
        self.btn_traceoutput.setChecked(self.params['traceoutput'])
        self.btn_traceoutput.setFixedWidth(setting_ui_width)
        self.btn_traceoutput.setFixedHeight(setting_ui_height)
        self.btn_traceoutput.clicked.connect(lambda: self.update_toggle_buttons())
        self.layout_form.addRow(QLabel('trace\noutput'), self.btn_traceoutput)

        self.edit_readsize = QLineEdit(str(self.params['readsize']))
        self.edit_readsize.setValidator(QDoubleValidator(0.0,  1000.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_readsize.setAlignment(Qt.AlignRight)
        self.edit_readsize.setFixedWidth(setting_ui_width)
        self.edit_readsize.setFixedHeight(setting_ui_height)
        self.layout_form.addRow(QLabel('read size\n(s)'), self.edit_readsize)

        self.btn_synthsndoutput = QPushButton('on' if self.params['synthsndoutput'] else 'off')
        self.btn_synthsndoutput.setCheckable(True)
        self.btn_synthsndoutput.setChecked(self.params['synthsndoutput'])
        self.btn_synthsndoutput.setFixedWidth(setting_ui_width)
        self.btn_synthsndoutput.setFixedHeight(setting_ui_height)
        self.btn_synthsndoutput.clicked.connect(lambda: self.update_toggle_buttons())
        self.layout_form.addRow(QLabel('synth snd\noutput'), self.btn_synthsndoutput)

        self.edit_mapL = QLineEdit(str(self.params['mapL']/1000.0))
        self.edit_mapL.setValidator(QDoubleValidator(0.0,  20.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_mapL.setAlignment(Qt.AlignRight)
        self.edit_mapL.setFixedWidth(int(setting_ui_width/2)-3)
        self.edit_mapL.setFixedHeight(setting_ui_height)

        self.edit_mapH = QLineEdit(str(self.params['mapH']/1000.0))
        self.edit_mapH.setValidator(QDoubleValidator(0.0,  20.0,  2, notation=QDoubleValidator.StandardNotation))
        self.edit_mapH.setAlignment(Qt.AlignRight)
        self.edit_mapH.setFixedWidth(int(setting_ui_width/2)-3)
        self.edit_mapH.setFixedHeight(setting_ui_height)

        layout_map = QHBoxLayout()
        layout_map.addWidget(self.edit_mapL)
        layout_map.addWidget(self.edit_mapH)
        self.layout_form.addRow(QLabel('map\n(kHz)'), layout_map)

        self.btn_proc_single = QPushButton('long file', self)
        self.btn_proc_single.clicked.connect(lambda: self.proc_single_file())
        self.btn_proc_multi = QPushButton('multiple files', self)
        self.btn_proc_multi.clicked.connect(lambda: self.proc_multi_file())
        self.btn_proc_stop = QPushButton('stop', self)
        self.btn_proc_stop.clicked.connect(lambda: self.proc_stop())
        self.btn_proc_stop.setEnabled(False)

        self.layout_setting.addWidget(self.btn_proc_single)
        self.layout_setting.addWidget(self.btn_proc_multi)
        self.layout_setting.addWidget(self.btn_proc_stop)

        self.Figure = plt.figure()
        self.FigureCanvas = FigureCanvas(self.Figure)
        self.layout_main.addLayout(self.layout_fig)
        self.label_filename = QLabel('File: file name')
        self.label_filename.setFixedHeight(20)
        self.layout_fig.addWidget(self.label_filename)
        self.layout_fig.addWidget(self.FigureCanvas)

        self.ax1 = self.Figure.add_subplot(3, 1, 1, xmargin=0, ymargin=0)
        self.ax2 = self.Figure.add_subplot(3, 1, 2, xmargin=0, ymargin=0)
        self.ax3 = self.Figure.add_subplot(3, 1, 3, xmargin=0, ymargin=0)
        self.Figure.tight_layout(pad=1.0)

        self.setLayout(self.layout_main)

        self.show()

    def update_toggle_buttons(self):

        self.update_params()
        self.btn_wavfileoutput.setText('on' if self.params['wavfileoutput'] else 'off')
        self.btn_imageoutput.setText('on' if self.params['imageoutput'] else 'off')
        self.btn_imagetype.setText('orig' if self.params['imagetype']==0 else 'flat')
        self.btn_traceoutput.setText('on' if self.params['traceoutput'] else 'off')
        self.btn_synthsndoutput.setText('on' if self.params['synthsndoutput'] else 'off')

    def enable_btns(self, flag):
        self.btn_proc_single.setEnabled(flag)
        self.btn_proc_multi.setEnabled(flag)
        self.btn_proc_stop.setEnabled(not flag)

    def show_progress(self, figdata):

        self.ax1.clear()
        self.ax1.set_xlabel('Time (s)')
        self.ax1.set_ylabel('Frequency (kHz)')
        self.ax1.margins(0.0)
        self.ax1.pcolormesh(figdata.x_disp, figdata.y_disp, figdata.mtsp_disp, cmap='Greys')

        self.ax2.clear()
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Frequency (kHz)')
        self.ax2.margins(0.0)
        self.ax2.pcolormesh(figdata.x_disp, figdata.y_disp, figdata.fltnd_disp, cmap='Greys')
        self.ax2.plot(np.array([np.min(figdata.x_disp), np.max(figdata.x_disp)]), 
                      np.array([figdata.freqmin/1000.0, figdata.freqmin/1000.0]), 'r--', linewidth=1.0)
        self.ax2.plot(np.array([np.min(figdata.x_disp), np.max(figdata.x_disp)]), 
                      np.array([figdata.freqmax/1000.0, figdata.freqmax/1000.0]), 'r--', linewidth=1.0)

        self.ax3.clear()
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Frequency (kHz)')
        self.ax3.margins(0.0)
        self.ax3.pcolormesh(figdata.x_disp, figdata.y_disp, figdata.fltnd_disp, cmap='Greys')

        for i_zone in range(figdata.nonsylzone.shape[0]):
            rect = matplotlib.patches.Rectangle((figdata.nonsylzone[i_zone, 0], np.min(figdata.fvec)/1000), 
                                                figdata.nonsylzone[i_zone, 1]-figdata.nonsylzone[i_zone, 0], 
                                                np.max(figdata.fvec)/1000-np.min(figdata.fvec)/1000, 
                                                alpha = 0.3, ec = 'none', fc = 'k', visible = True)
            self.ax3.add_patch(rect)
        
        for i_zone in range(figdata.margzone.shape[0]):
            rect = matplotlib.patches.Rectangle((figdata.margzone[i_zone, 0], np.min(figdata.fvec)/1000), 
                                                figdata.margzone[i_zone, 1]-figdata.margzone[i_zone, 0], 
                                                np.max(figdata.fvec)/1000-np.min(figdata.fvec)/1000, 
                                                alpha = 0.1, ec = 'none', fc = 'k', visible = True)
            self.ax3.add_patch(rect)

        
        self.ax3.plot(figdata.tvec, figdata.freqtrace[:,:3]/1000.0, 'b.', markersize=3)

        if self.worker.flag_multi:
            n = len(self.worker.fp)
            i_file = self.worker.i_file
            self.label_filename.setText(figdata.info_str + ' [{:d}/{:d} files]'.format(i_file+1, n))
        else:
            self.label_filename.setText(figdata.info_str)

        self.label_fs.setText('{:d}'.format(int(figdata.fs/1000.0)))

        self.Figure.tight_layout(pad=1.0)

        self.FigureCanvas.draw()
        self.update()

    def proc_single_file(self):
        
        self.update_params()

        fp, _ = QFileDialog.getOpenFileName(self, 'input file', '.', filter='(*.wav *.flac)')
        if len(fp) == 0:
            return 
        
        savefp, _ = QFileDialog.getSaveFileName(self, 'save file name', os.path.splitext(fp)[0] + '_dat.csv', filter='*.csv')
        if len(savefp) == 0:
            return 
        
        if self.params['synthsndoutput']:
            fname_audiblewav, _ = QFileDialog.getSaveFileName(self, 'save synth snd file name', os.path.splitext(fp)[0] + '_syn.wav', filter='*.wav')
            if len(fname_audiblewav) == 0:
                return
        else:
            fname_audiblewav = None

        outp = QFileDialog.getExistingDirectory(self, 'output directory', os.path.dirname(fp))
        if len(outp) == 0:
            return 

        self.thread = QThread() 
        self.worker = ProcWorker(False, self.params, fp, savefp, outp, fname_audiblewav) 
        self.worker.moveToThread(self.thread) 

        self.thread.started.connect(self.worker.run) 
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.show_progress)

        self.thread.start() 

        self.enable_btns(False)
        self.thread.finished.connect(
            lambda: self.enable_btns(True)
        )
        self.thread.finished.connect(
            lambda: QMessageBox.information(None, 'info', 'Done!', QMessageBox.Ok)
        )

        
    def proc_multi_file(self):
        
        self.update_params()

        indir = QFileDialog.getExistingDirectory(self, 'input directory', '.')
        if len(indir) == 0:
            return 
        
        fp = []
        for f in glob.glob(indir + '/*.wav'):
            if '_syn.wav' not in f:
                fp.append(f)

        if len(fp) == 0:
            return
        
        self.thread = QThread() 
        self.worker = ProcWorker(True, self.params, fp) 
        self.worker.moveToThread(self.thread) 

        self.thread.started.connect(self.worker.run) 
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.show_progress)

        self.thread.start() 

        self.enable_btns(False)
        self.thread.finished.connect(
            lambda: self.enable_btns(True)
        )
        self.thread.finished.connect(
            lambda: QMessageBox.information(None, 'info', 'Done!', QMessageBox.Ok)
        )

    def proc_stop(self):
        if self.worker is not None:
            if self.worker.running:
                self.thread.requestInterruption()
                self.thread.quit()
                self.thread.wait()
                print('The processing was interrupted.')

    def update_params(self):

        self.params['timestep'] = float(self.edit_timestep.text())/1000.0
        self.params['freqmin'] = float(self.edit_freqmin.text())*1000.0
        self.params['freqmax'] = float(self.edit_freqmax.text())*1000.0
        self.params['threshval'] = float(self.edit_thresh.text())
        self.params['durmin'] = float(self.edit_durmin.text())/1000.0
        self.params['durmax'] = float(self.edit_durmax.text())/1000.0
        self.params['gapmin'] = float(self.edit_gapmin.text())/1000.0
        self.params['margin'] = float(self.edit_margin.text())/1000.0

        self.params['wavfileoutput'] = self.btn_wavfileoutput.isChecked()
        self.params['imageoutput'] = self.btn_imageoutput.isChecked()
        self.params['imagetype'] = int(self.btn_imagetype.isChecked())
        self.params['wavfileoutput'] = self.btn_wavfileoutput.isChecked()
        self.params['traceoutput'] = self.btn_traceoutput.isChecked()

        self.params['readsize'] = float(self.edit_readsize.text())
        self.params['mapL'] = float(self.edit_mapL.text())*1000.0
        self.params['mapH'] = float(self.edit_mapH.text())*1000.0
        self.params['synthsndoutput'] = self.btn_synthsndoutput.isChecked()
        
        with open(prm_file_path, 'w') as f:
            yaml.dump(self.params, f)

    def closeEvent(self, event):

        self.proc_stop()

        self.update_params()

        event.accept() # let the window close

def main():
    app = QApplication(sys.argv)
    mwin = MainWindow()
    sys.exit(app.exec_())

if __name__ == "__main__":

    main()