import sys
import random
import json
import joblib
import time
import os
import matplotlib
from datetime import datetime
from matplotlib import colors
import pandas as pd
import numpy as np
import matplotlib.ticker as mticker
from paho.mqtt import client as mqtt_client
from library.utility import temporal
from publish import publish
from collections import deque
from threading import Thread
from multiprocessing import Process
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Qt5Agg")

# Window untuk GUI
class CustomMainWindow(QMainWindow):
    def __init__(self):
        global app, myGUI
        super(CustomMainWindow, self).__init__()
        self.setGeometry(400, 400, 800, 800)
        self.setWindowTitle("DETEKSI STICTION")

        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

        self.FRAME_A = QFrame(self)
        self.FRAME_A.setStyleSheet(f"background-color: {QColor(50, 50, 50).name()};" )
        self.FRAME_A.setGeometry(QRect(400, 400, 800, 800))
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)
        self.myFig = CustomFigCanvas()
        self.LAYOUT_A.addWidget(self.myFig, *(2,0))

        self.title = QLabel('      REALTIME VALVE STICTION DETECTION')
        self.title.setFixedHeight(30)
        self.title.setFont(QFont('Times', 16, weight = QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet(f"color: {QColor(255, 255, 255).name()};")
        self.LAYOUT_A.addWidget(self.title, *(1,0))
        
        self.PV_value = 0
        self.OP_value = 0
        self.label = '-'
        self.title = QLabel(f'      PV : {self.PV_value}           OP : {self.OP_value}              STATUS : {self.label}')
        self.title.setFixedHeight(30)
        self.title.setFont(QFont('Times', 12, weight = QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet(f"color: {QColor(255, 255, 255).name()};")
        self.LAYOUT_A.addWidget(self.title, *(3,0))
        
        # Thread 1 untuk PV
        PV_stream = Thread(name = 'PV_stream', target = com_PV.signal.connect, daemon = True, args = (self.PV_callback,))
        PV_stream.start()

        # Thread 2 untuk OP
        OP_stream = Thread(name = 'OP_stream', target = com_OP.signal.connect, daemon = True, args = (self.OP_callback,))
        OP_stream.start()

        # Thread 3 untuk Valve detection
        detection_stream = Thread(name = 'detection_stream', target = com_detection.signal.connect, daemon = True, args = (self.detection_callback,))
        detection_stream.start()
        a = 0

        self.show()

        run_sub = Thread(name = 'subscribe', target = main().run, daemon=True)
        run_sub.start()
        return

    # PV_callback untuk PV
    def PV_callback(self, value):
        self.myFig.add_PV(value)
        self.PV_value = '{:.3f}'.format(value)
        return

    #OP_callback untuk OP
    def OP_callback(self, value):
        self.myFig.add_OP(value)
        self.OP_value = '{:.3f}'.format(value)
        return

    #detection_callback untuk Valve detection
    def detection_callback(self, value):
        self.myFig.add_detection(value)
        if value == 1 :
            self.label = 'STICTION'
        elif value == 0 :
            self.label = 'NO STICTION'
        else :
            self.label = 'DATA < 100'
        self.title.setText(f'       PV : {self.PV_value}           OP : {self.OP_value}              STATUS : {self.label}')
        return

''' End Class '''

# Plot realtime
class CustomFigCanvas(FigureCanvas, TimedAnimation):
    def __init__(self):
        global app, myGUI
        self.PV_data = [] #PV
        self.OP_data = [] #OP
        self.detection_data = [] #Valve detection

        self.PV_count = 1
        self.OP_count = 1
        self.detection_count = 1
        # self.OP_count = 0

        self.PV_lim = deque(maxlen = 103)
        self.OP_lim = deque(maxlen = 103)
        self.time_lim = 103 #Valve detection time axis

        self.time = np.linspace(0, self.time_lim - 1, self.time_lim) #Tick Valve detection time axis
        self.PV = (self.time * 0.0) #PV axis
        self.OP = (self.time * 0.0) #OP axis
        self.detection = (self.time * 0.0) - 1 #Valve detection axis

        # The window
        self.fig, self.ax = plt.subplots(3)
        self.fig.set_facecolor((50/255, 50/255, 50/255))
        self.PV_ax = self.ax[0] #window 1 PV
        self.OP_ax = self.ax[1] #window 2 OP
        self.detection_ax = self.ax[2] #windows 3 valve detection
        self.fig.set_tight_layout(True)

        # PV plot setting
        self.PV_ax.set_ylabel('PV', color = (1, 1, 1), fontweight = 'bold', fontsize = 12)
        self.PV_ax.set_title('PV - Sample', color = (1, 1, 1), fontweight = 'bold', fontsize = 12)
        self.PV_line = Line2D([], [], color=(1, 1, 0), linewidth = 2)
        self.PV_line_tail = Line2D([], [], color=(1, 1, 204/255), linewidth=2)
        self.PV_line_head = Line2D([], [], color=(1, 1, 204/255), marker='o', markeredgecolor='r')
        self.PV_ax.add_line(self.PV_line)
        self.PV_ax.add_line(self.PV_line_tail)
        self.PV_ax.add_line(self.PV_line_head)
        self.PV_ax.set_xlim(0, self.time_lim - 1)
        self.PV_ax.set_xticklabels([])
        self.PV_ax.set_facecolor((50/255, 50/255, 50/255))
        self.PV_ax.spines[:].set_color((1, 1, 1))
        self.PV_ax.spines[:].set_linewidth(2)
        self.PV_ax.tick_params(colors=(1, 1, 1), width = 2, labelcolor = (1, 1, 1))
        
        # OP plot setting
        self.OP_ax.set_ylabel('OP', color = (1, 1, 1), fontweight = 'bold', fontsize = 12)
        self.OP_ax.set_title('OP - Sample', color = (1, 1, 1), fontweight = 'bold', fontsize = 12)
        self.OP_line = Line2D([], [], color=(1, 178/255, 102/255), linewidth=2)
        self.OP_line_tail = Line2D([], [], color=(1, 204/255, 153/255), linewidth=2)
        self.OP_line_head = Line2D([], [], color=(1, 204/255, 153/255), marker='o', markeredgecolor='r')
        self.OP_ax.add_line(self.OP_line)
        self.OP_ax.add_line(self.OP_line_tail)
        self.OP_ax.add_line(self.OP_line_head)
        self.OP_ax.set_xlim(0, self.time_lim - 1)
        self.OP_ax.set_xticklabels([])
        self.OP_ax.set_facecolor((50/255, 50/255, 50/255))
        self.OP_ax.spines[:].set_color((1, 1, 1))
        self.OP_ax.spines[:].set_linewidth(2)
        self.OP_ax.tick_params(colors=(1, 1, 1), width = 2, labelcolor = (1, 1, 1))

        # Valve detection plot setting
        self.detection_ax.set_xlabel('Sample', color = (1, 1, 1), fontweight = 'bold', fontsize = 12)
        self.detection_ax.set_title('Valve Condition - Sample', color = (1, 1, 1), fontweight = 'bold', fontsize = 12)
        self.detection_line = Line2D([], [], color=(1, 153/255, 153/255), linewidth=2)
        self.detection_line_tail = Line2D([], [], color=(1, 204/255, 204/255), linewidth=2)
        self.detection_line_head = Line2D([], [], color=(1, 153/255, 153/255), marker='o', markeredgecolor='r')
        self.detection_ax.add_line(self.detection_line)
        self.detection_ax.add_line(self.detection_line_tail)
        self.detection_ax.add_line(self.detection_line_head)
        self.detection_ax.set_xlim(0, self.time_lim - 1)
        self.detection_ax.set_ylim(-1.5, 1.5)
        self.detection_ax.set_xticks(self.detection_ax.get_xticks()[:-1])
        self.detection_ax.set_xticklabels([])
        self.detection_ax.set_yticks(self.detection_ax.get_yticks()/2)
        self.detection_ax.set_yticklabels(['data < 100', '', 'No stiction', '', 'Stiction'])
        self.detection_ax.set_facecolor((50/255, 50/255, 50/255))
        self.detection_ax.spines[:].set_color((1, 1, 1))
        self.detection_ax.spines[:].set_linewidth(2)
        self.detection_ax.tick_params(colors=(1, 1, 1), width = 2, labelcolor = (1, 1, 1))

        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 50, blit = False)
        return

    #frame sequence
    def new_frame_seq(self):
        return iter(range(self.time.size))
    
    #add data PV
    def add_PV(self, value):
        self.PV_lim.append(value)
        self.PV_data.append(value)
        return

    #add data OP
    def add_OP(self, value):
        self.OP_lim.append(value)
        self.OP_data.append(value)
        # print(self.OP_lim)
        # print(min(self.OP_lim), max(self.OP_lim)) 
        return

    #add data valve detection
    def add_detection(self, value):
        self.detection_data.append(value)
        return
        
    #plot step
    def _step(self, *args):
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass
        return

    #draw animation plot
    def _draw_frame(self, framedata):
        margin = 2
        #PV
        while(len(self.PV_data) > 0):
            self.PV = np.roll(self.PV, -1)
            self.PV[-3] = self.PV_data[0]    
            del(self.PV_data[0])
            PV_low = min(self.PV_lim)
            PV_up = max(self.PV_lim)
            self.PV_ax.set_ylim(PV_low, PV_up)
            self.PV_count += 1
                    
        # print(len(self.time[ 0 : self.time.size - margin ]))#, len(np.append(self.time[-10:-1 - margin], self.time[-1 - margin]), len(self.time[-1 - margin])))
        self.PV_line.set_data(self.time[ 0 : self.time.size - margin ], self.PV[ 0 : self.time.size - margin ])
        self.PV_line_tail.set_data(self.time[-10:- margin], self.PV[-10: -margin])#np.append(self.time[-10:-1 - margin], self.time[-1 - margin]), np.append(self.PV[-10:-1 - margin], self.PV[-1 - margin]))
        self.PV_line_head.set_data(self.time[-1 - margin], self.PV[-1 - margin])

        #OP
        while(len(self.OP_data) > 0):
            self.OP = np.roll(self.OP, -1)
            self.OP[-3] = self.OP_data[0]    
            del(self.OP_data[0])
            OP_low = min(self.OP_lim)
            OP_up = max(self.OP_lim)
            self.OP_ax.set_ylim(OP_low, OP_up)
            self.OP_count += 1
            print(self.OP_ax.get_ylim(), min(self.OP_lim), max(self.OP_lim))
                    
        # print(len(self.time[ 0 : self.time.size - margin ]))#, len(np.append(self.time[-10:-1 - margin], self.time[-1 - margin]), len(self.time[-1 - margin])))
        self.OP_line.set_data(self.time[ 0 : self.time.size - margin ], self.OP[ 0 : self.time.size - margin ])
        self.OP_line_tail.set_data(self.time[-10:- margin], self.OP[-10: -margin])#np.append(self.time[-10:-1 - margin], self.time[-1 - margin]), np.append(self.OP[-10:-1 - margin], self.OP[-1 - margin]))
        self.OP_line_head.set_data(self.time[-1 - margin], self.OP[-1 - margin])

        #Valve detection
        while(len(self.detection_data) > 0):
            self.detection = np.roll(self.detection, -1)
            self.detection[-3] = self.detection_data[0]
            del(self.detection_data[0])
            self.detection_ax.set_xticklabels([-100 + self.detection_count, -80 + self.detection_count, -60 + self.detection_count, -40 + self.detection_count, -20 + self.detection_count, self.detection_count])
            self.detection_count += 1
            
        self.detection_line.set_data(self.time[ 0 : self.time.size - margin ], self.detection[ 0 : self.time.size - margin ])
        self.detection_line_tail.set_data(self.time[-10:- margin], self.detection[-10: -margin])#np.append(self.time[-10:-1 - margin], self.time[-1 - margin]), np.append(self.detection[-10:-1 - margin], self.detection[-1 - margin]))
        self.detection_line_head.set_data(self.time[-1 - margin], self.detection[-1 - margin])

        #tampilkan 3 plot 
        self._drawn_artists = [self.PV_line, self.PV_line_tail, self.PV_line_head, self.OP_line, self.OP_line_tail, self.OP_line_head, self.detection_line, self.detection_line_tail, self.detection_line_head]
        return

''' End Class '''

#data communication
class Communicate(QObject):
    signal = pyqtSignal(float)

''' End Class '''

class main():
    def __init__(self):
        self.broker = 'localhost'
        self.port = 1883
        self.client_id = f'python-mqtt-{random.randint(0, 100)}'
        self.topic = "Data"
        self.data = pd.DataFrame()
        self.dat = pd.DataFrame()
        self.model = joblib.load(r'model akhir.joblib') #load model ML yang telah disimpan
        self.extract = temporal(n = 100, progress=False)
        #features yang digunakan -> didapatkan dari step pembuatan model ML
        self.features =['PV Absolute Energy',
                        'PV Entropy',
                        'PV Mean Absolute Differences',
                        'PV Mean Differences',
                        'PV Median Differences',
                        'PV Negative Turning Point',
                        'PV Neighbourhood Peaks',
                        'PV Slope',
                        'PV Zero Crossing Rate',
                        'OP Absolute Energy',
                        'OP Entropy',
                        'OP Mean Absolute Differences',
                        'OP Mean Differences',
                        'OP Median Differences',
                        'OP Negative Turning Point',
                        'OP Neighbourhood Peaks',
                        'OP Peak to Peak Distance',
                        'OP Slope',
                        'OP Zero Crossing Rate']
        self.label = -1
        self.hasil_deteksi = pd.DataFrame(columns = ['Send Time', 'Received Time', 'Detection Time', 'ID'] + self.features + ['Prob No', 'Prob Yes', 'Prediksi', 'Aktual'])
        self.data_deteksi = pd.DataFrame(columns= ['Send Time', 'Received Time', 'Detection Time', 'ID', 'PV', 'OP', 'Prob No', 'Prob Yes', 'Prediksi', 'Aktual'])
        self.count = 1
        self.id = ['nan']
        if os.path.isdir(r'hasil deteksi'):
            pass
        else :
            os.mkdir('hasil deteksi')

        if os.path.exists('hasil deteksi/hasil deteksi.csv'):
            pass
        else :
            self.hasil_deteksi.to_csv('hasil deteksi/hasil deteksi.csv', index=False)
 
        if os.path.exists('hasil deteksi/data deteksi.csv') :
            pass
        else :
            self.data_deteksi.to_csv('hasil deteksi/data deteksi.csv', index = False)

    #connect MQTT
    def connect_mqtt(self) -> mqtt_client:
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Subscriber Connected to MQTT Broker!")
            else:
                print("Subscriber Failed to connect, return code %d\n", rc)

        global client_id, broker, port
        client = mqtt_client.Client(self.client_id)
        client.on_connect = on_connect
        client.connect(self.broker, self.port)
        return client

    #subscribe image from MQTT
    def on_message(self, client, userdata, msg):
        df = msg.payload.decode() #decode pesan yang diterima
        df2 = pd.json_normalize(json.loads(df)) #pesan yang diterima berupa jason --> pandas dataframe
        df2.insert(1, 'Received Time', pd.Timestamp.now())

        pv = df2.PV.values[0] #nilai pv
        op = df2.OP.values[0] #nilai op

        if 'Label' not in df2.columns :
            df2['Label'] = 'no label'

        rec_time = df2['Received Time'].values[0]

        if 'Send Time' not in df2.columns :
            send_time = '-'
        else :
            send_time = df2['Send Time'].values[0]

        actual_label = df2.Label.values[0]
        id = df2.ID.values[0]
        self.id.append(id)

        self.data = pd.concat([self.data, df2]).reset_index(drop=True) #pesan yg diterima per baris, disatukan dengan pd.concat

        print(f"{self.count} Received `{df}` from `{msg.topic}")
        self.count += 1
        
        if len(self.data) < 100 :
            raw_saved = pd.read_csv('hasil deteksi/data deteksi.csv')
            raw_new = pd.DataFrame([send_time, rec_time, '-', id, pv, op, '-', '-', '-', actual_label], index = raw_saved.columns).T
            raw_new = pd.concat([raw_saved, raw_new])
            raw_new.to_csv('hasil deteksi/data deteksi.csv', index = False)

        else : #Ambil tiap 100 data untuk diprediksi
            self.data = self.data.loc[:,['PV', 'OP']] #ambil nilai PV, OP saja
            
            data_saved = pd.read_csv('hasil deteksi/hasil deteksi.csv')

            #timeseries feature extraction
            data2 = self.extract.extract(self.data)[self.features]

            #predict classification valve condition
            data2 = data2.values
            self.label = self.model.predict(data2)[0]

            prob = self.model.predict_score(data2)[0]
            # print(prob)
            yes_prob = prob[1]
            no_prob = prob[0]

            dec_time = str(pd.Timestamp.now())

            col = data_saved.columns
            data2_list = [send_time, rec_time, dec_time, id] + data2[0].tolist() + [no_prob, yes_prob, self.label, actual_label]
            data_new = pd.DataFrame(data2_list, index = col).T
            data_new = pd.concat([data_saved, data_new])
            data_new.to_csv('hasil deteksi/hasil deteksi.csv', index = False)

            raw_saved = pd.read_csv('hasil deteksi/data deteksi.csv')
            raw_new = pd.DataFrame([send_time, rec_time, dec_time, id, pv, op, no_prob, yes_prob, self.label, actual_label], index = raw_saved.columns).T
            raw_new = pd.concat([raw_saved, raw_new])
            raw_new.to_csv('hasil deteksi/data deteksi.csv', index = False)
            #buang data ke-1 pada dataframe untuk kemudian menerima 1 data baru di akhir dataframe
            self.data = self.data.iloc[1:,:]

        com_PV.signal.emit(pv) #kirim data pv ke plot
        com_OP.signal.emit(op) #kirim data op ke plot
        com_detection.signal.emit(self.label) #kirim label
        
    #untuk run mqtt
    def run(self):
        client = self.connect_mqtt()
        client.subscribe(self.topic, qos = 1)
        client.on_message = self.on_message
        client.loop_forever()

    def start(self):
        app = QApplication(sys.argv)
        QApplication.setStyle(QStyleFactory.create('Plastique'))
        myGUI = CustomMainWindow()
        sys.exit(app.exec_())

# global com_PV, com_OP, com_detection
com_PV = Communicate()
com_OP = Communicate()
com_detection = Communicate()

if __name__ == "__main__":
    main().start()