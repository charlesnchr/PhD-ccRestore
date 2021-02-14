import glob
import os

# folders = glob.glob('G:/My Drive/01models/SIMRec/december-revision/*')
folders = []
folders.append("G:/My Drive/01models/SIMRec/february/0216_SIMRec_0214_rndAll_rcan_continued")
folders.append("G:/My Drive/01models/SIMRec/march/0312_SIMRec_rndAll_0309")
basecmd = 'python run.py'
baseout = 'C:/Users/Charles/Desktop/Lisa-out-20210203'
workdir = '.'

os.makedirs(baseout,exist_ok=True)

newopts = {
    'root': '"G:/My Drive/01datasets/SIMRec/Lisa/Lisa_new atheisim"',
    'out': baseout + '/%s',
    'sourceimages_path':'',
    'datagen_workers':'',
    'nrep':'',
    'NoiseLevel':'',
    'NoiseLevelRandFac':'',
    'Nangle':'',
    'Nshift':'',
    'phaseErrorFac':'',
    'alphaErrorFac':'',
    'disposableTrainingData':'',
    'cloud':'',
    'usePoissonNoise':'',
    'dontShuffleOrientations':'',
    'applyOTFtoGT':'',
    'k2':'',
    'k2_err':'',
    'PSFOTFscale':'',
    'ModFac':'',
    'nplot':0,
    'ntest':200
}


cmds = []

for folder in folders:
    if '3x5' in folder or '5x5' in folder:
        continue
    log = open('%s/log.txt' % folder,'r').readlines()
    
    argsFound = False
    
    for line in log:
        if 'ARGS:' in line:
            argsFound = True
            line = line.replace('\n','')
            opts = line.split(' --')[1:]
            
            newoptarray = []
            
            for opt in opts:
                optname = opt.split(' ')[0]
                if optname in newopts:
                    if newopts[optname] != '':
                        newoptarray.append('%s %s' % (optname,newopts[optname]))
                else:
                    newoptarray.append(opt)
            
            optstring = ' --'.join(newoptarray)
            
            if '20201222' in folder:
                optstring = optstring % 'lowerr_highnfeat'
            else:
                optstring = optstring % os.path.basename(folder).split('_')[-1]
                            
            cmds.append('%s --%s %s' % (basecmd,optstring,'--weights "%s/final.pth" --test' % folder))
    
    if argsFound == False:
        
        if 'february' in folder:
            otheropts = '--n_resblocks 10 --n_resgroups 3 --n_feat 48 --norm hist'
        else:
            otheropts = '--n_resblocks 10 --n_resgroups 3 --n_feat 96 --norm hist'
            
        outname = os.path.basename(folder)

        cmd = '''
            python run.py
            --root %s
            --out %s
            --scale 1
            --task simin_gtout
            --model rcan
            --nch_in 9
            --nch_out 1
            --dataset fouriersim
            --weights "%s"
            --ntrain 1
            --ntest 200
            --test 
            %s
            ''' % (newopts['root'],newopts['out'] % outname,'%s/final.pth' % folder,otheropts) 

        cmds.append(cmd.replace('\n',' ').replace('\\','/'))



# fid = open('cmds.sh','w')
# fid.write('#!/bin/bash\n')
# fid.write('cd ccRestore\n')
fid = open('cmds.bat','w')

if len(cmds) > 0:
    for cmd in cmds:
        fid.write(cmd + '\n')
else:
    fid.write('echo no commands')