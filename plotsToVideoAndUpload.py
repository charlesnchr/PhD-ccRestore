import numpy as np
import skimage.io
import skvideo.io
import glob
import os
import sys

def procfolder(basedir,uploaddir):

    ntest = 0

    try:
        log = open('%s/log.txt' % basedir,'r')
        ns = log.readline()
        ntest = int(ns.split('ntest=')[1].split(',')[0])
    except:
        print('could not open log file',basedir)
        return


    files = glob.glob(basedir + '/*.png')
    if len(files) == 0:
        print('previously processed',basedir,'- skipping')
        return


    for i in range(ntest):
        files = glob.glob(basedir + '/combined_epoch*_%d.png' % i)

        if len(files) == 0: continue

        filename = '%s/%d.mp4' % (basedir,i)
        print('\nBuilding',filename)

        fps = 5
        writer = skvideo.io.FFmpegWriter(filename,inputdict={'-r':str(fps)},outputdict={'-r':str(fps),'-crf':'0'})

        count = 0
        for j,file in enumerate(files):
            I = skimage.io.imread(file)
            writer.writeFrame(I)
            print(' \r [%d/%d]' % (j+1,len(files)),end='')
            count += 1

        if count == len(files):
            for file in files:
                os.remove(file)
        
        writer.close()
    
    # upload
    os.system('onedrivecmd put %s %s' % (basedir,uploaddir))


basedir = sys.argv[1]
uploaddir = sys.argv[2]

print('basedir',basedir)

logfile = glob.glob('%s/log.txt' % basedir)

if len(logfile) == 0: # try iterating folders
    folders = [name for name in glob.glob('%s/*' % basedir) if os.path.isdir(name)]
    print('going over folders:',folders)
    for folder in folders:
        logfile = glob.glob('%s/log.txt' % (folder))
        if len(logfile) > 0:
            procfolder('%s' % folder,uploaddir)
        else:
            print('log file not found for',folder)
else:
    print('log file found, processing folder',basedir)
    procfolder(basedir,uploaddir)
