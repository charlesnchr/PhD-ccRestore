import subprocess
import time
import threading 

count = 0

while True:
    gpustat = subprocess.check_output('gpustat')
    print('we have',gpustat)
    gpustat = str(gpustat).split('\\n')
    print('after string conversion',gpustat)
    for idx,line in enumerate(gpustat):
        if '|' not in line: continue
        idx = line.split(']')[0].split('[')[-1]
        if line.split('|')[-1].isspace():
            print('gpu',idx,'is free')
        if 'jm2311' in line:
            print('gpu',idx,'occupied by jm2311')
        if 'cnc39' in line:
            print('gpu',idx,'by me')
        
        memory_used = int(line.split('/')[0].split('|')[-1].strip())
        memory_total = int(line.split('/')[1].split('MB')[0].strip())

        print('gpu',idx,'has',memory_used,memory_total,'ratio',memory_used/memory_total)

    time.sleep(2)
    count += 1
    if count == 3:
        print('now running job')
        subprocess.Popen('python gputest.py', shell=True)
        print('now started command! waiting a bit')
        time.sleep(20)
    
    if count == 4:
        'now quitting'
        break

    print('\n\n')
        