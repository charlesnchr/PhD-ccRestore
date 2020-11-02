from flask import Flask
import requests
import subprocess
import time
import threading 

app = Flask(__name__)

def reportGPU():
    gpustat = subprocess.check_output('gpustat')
    requests.post('http://82.0.79.244', data={'cmddodsuto':'nothing','gpustat':gpustat})

def threadWatch():
    while True:
        reportGPU()
        time.sleep(0.5)

if __name__ == '__main__':
    t = threading.Thread(target=threadWatch)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', debug=True)
