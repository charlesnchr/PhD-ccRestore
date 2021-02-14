import glob
import os

maincmd = '''
python run.py --root "%s" --out "%s" --scale 1 --task simin_gtout --model rcan --nch_in 9 --nch_out 1 --dataset fouriersim --weights "/Volumes/GoogleDrive/My Drive/01models/SIMRec/march/0312_SIMRec_rndAll_0309/final.pth" --ntrain 1 --ntest 200 --test  --n_resblocks 10 --n_resgroups 3 --n_feat 96 --norm hist --cpu
'''

rootfolder = "/Volumes/GoogleDrive/My Drive/01datasets/SIMRec/Lisa/September 2020"
outfolder = "/Volumes/GoogleDrive/My Drive/0main/0phd/plots-for-Lisa-2021/All_reconstructions_from_best_model/Lisa (Sep 2020)"

fid = open("cmds.sh",'w')
files = glob.glob("%s/**/*.tif" % rootfolder,recursive=True)

for file in files:
    if 'SIM.tif' in file or 'Widefield.tif' in file: continue

    newpath = file.replace(rootfolder,outfolder)
    pardir = os.path.abspath(os.path.join(newpath, ".."))
    cmd = maincmd % (file, pardir)
    print(cmd,end='\n')
    fid.write(cmd)
