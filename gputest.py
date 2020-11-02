

import torch
import time


x = torch.ones((1000,1000)).cuda()

print('initialised')

time.sleep(10)

print('slept 10 secs, quitting')