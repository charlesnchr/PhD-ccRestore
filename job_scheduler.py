import subprocess
import time
import threading 
import sys

max_concurrent_jobs = 2
count = 0

if len(sys.argv) < 2:
    print('provide command line argument with a jobscript file')
    sys.exit(0)

if len(sys.argv) > 2:
    max_concurrent_jobs = int(sys.argv[2])

jobscript = sys.argv[1]
t0 = time.perf_counter()
jobs = open(jobscript,'r').read().split('\n')
jobs = [job for job in jobs if job.strip() != ''] # remove empty lines
submission_times = {}

print('Given %d jobs' % len(jobs))

while True:
    gpustat = subprocess.check_output('gpustat')
    gpustat = str(gpustat).split('\\n')

    # check current jobs
    current_jobs = 0
    for idx,line in enumerate(gpustat):
        if '|' not in line: continue
          
        idx = line.split(']')[0].split('[')[-1]
        
        if 'cnc39' in line:
            # print('\ngpu',idx,'used by me')
            current_jobs += 1

    if current_jobs < max_concurrent_jobs:


        # check for available gpu
        submit_job = None
        for idx,line in enumerate(gpustat):
            if '|' not in line: continue
            idx = line.split(']')[0].split('[')[-1]

            # mem could be used to squeeze on only mildly busy GPUs
            memory_used = int(line.split('/')[0].split('|')[-1].strip())
            memory_total = int(line.split('/')[1].split('MB')[0].strip())
            # print('\ngpu',idx,'has',memory_used,memory_total,'ratio',memory_used/memory_total)

            # check if prevoiusly recently used
            if submission_times.get(idx) is not None:
                tnow = time.perf_counter()
                if tnow - submission_times.get(idx) < 1800: # wait at least 30 minutes
                    continue

            if len(line.split('|')[-1].strip()) == 0 or memory_used < 500:
                submit_job = idx
                break
        
        if submit_job is not None:
            print('\nGPU %s to run job: %s' % (idx,jobs[count]))
            submission_times[idx] = time.perf_counter()
            subprocess.Popen('CUDA_VISIBLE_DEVICES=%s %s' % (idx,jobs[count]), shell=True, stdout=subprocess.DEVNULL)
            count += 1
            print('\nnow started command!')
            
            if count == len(jobs):
                print('\nsubmitted all jobs')
                break
            
            print('\nWaiting 10 seconds')
            time.sleep(10)


    # extra wait
    time.sleep(2)
    tdiff = time.perf_counter() - t0
    print('status: submitted %d jobs out of %d, currently running: %d, time spent %0.1f s' % (count,len(jobs),current_jobs,tdiff),end='\r')
    
print('\nscheduler exiting')