import torch
import time
import sys

idx = sys.argv[1]
x = torch.ones((1000,1000)).cuda()

print('initialised')

time.sleep(120)

print('slept 90 secs, quitting')

open('testlog' + idx + '.txt','w').write('hi from job %s' % idx)