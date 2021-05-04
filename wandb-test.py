import wandb
import argparse
import time
from skimage import data
import numpy as np

wandb.init(project="phd")

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                     help='input batch size for training (default: 8)')
args = parser.parse_args()
wandb.config.update(args) # adds all of the arguments as config variables

for i in range(50):
    time.sleep(1)
    wandb.log({'accuracy': np.random.rand()+i, 'loss': np.random.rand()/(i+1), 'img': [wandb.Image(data.astronaut()/255*i/50)]})


