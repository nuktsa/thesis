import random
import time
import pandas as pd
from paho.mqtt import client as mqtt_client
import os, glob
from datetime import datetime

class publish():
    def __init__(self):
        self.broker = 'localhost'
        self.port = 1883
        self.topic = "Data"
        self.client_id = f'python-mqtt-{random.randint(0, 1000)}'

    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)

        global client_id, broker, port
        client = mqtt_client.Client(self.client_id)
        client.on_connect = on_connect
        client.connect(self.broker, self.port)
        return client

    def publish(self, client):
        msg_count = 0
        
        path = r'Data\Data Validasi SACAC' #path kumpulan csv kalang kontrol
        no = glob.glob(path + r'\no\*csv')
        yes = glob.glob(path + r'\yes\*csv')
        file = yes[6] #buat ganti2 file yang mau dikirim, yes[6] -> file yes ke 6. ganti no jika mau file no
        label = 1 #yes -> label = 1, no -> label = 0, untuk demo saja
        
        data = pd.read_csv(file) #data yg akan dipublish
        data.insert(0, 'ID', os.path.basename(file)[:-4])
        if label :
            data['Label'] = label
        else :
            data['Label'] = 'No Label'

        for i in range (len(data)) :
            time.sleep(0.5) #disesuaikan dengan time sampling sensor, untuk demo dapat dipercepat
            df = pd.concat([pd.Series({'Send Time' : str(pd.Timestamp.now())}), data.loc[i]])
            msg = df.to_json()
            result = client.publish(self.topic, msg)
            status = result[0]
            if status == 0:
                print(f"`{msg}` send to `{self.topic}`")
            else:
                print(f"Failed to send message to topic {self.topic}")
            msg_count += 1

    def start(self):
        client = self.connect_mqtt()
        client.loop_start()
        self.publish(client)

if __name__ == "__main__":
    publish().start()