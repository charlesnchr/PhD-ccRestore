#!/bin/bash
: ' ----------------------------------------
* Creation Time : Thu 15 Jun 2023 01:13:32 BST
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------'

modfac_values=(0.1 0.2 0.3 0.4)

# Array of PSFOTFscale values
psf_values=(0.4 0.5 0.6 0.7 0.8 0.9)

# Iterate over ModFac values
for modfac in "${modfac_values[@]}"
do
    # Iterate over PSFOTFscale values
    for psf in "${psf_values[@]}"
    do
        # Run your command
        python MLSIM_pipeline.py --root auto --sourceimages_path ~/2work/onirepos/SIM/training_data/DIV2K --nrep 1 --datagen_workers 0 --imageSize 684 428 --out "runs/$(date +"%Y%m%d_%H%M%S")-rcan-modfac-${modfac}-psf-${psf}" --model rcan --nch_in 9 --nch_out 1 --ntrain 2 --ntest 1 --scale 2 --task simin_gtout --n_resgroups 3 --n_resblocks 10 --n_feats 64 --lr 0.0001 --nepoch 100 --scheduler 10,0.5 --norm minmax --dataset fouriersim --batchSize 2 --saveinterval 10 --plotinterval 10 --nplot 5 --Nangle 3 --Nshift 3 --k2 100 --dataonly --disable_wandb --SIMmodality spots --dmdMapping 2 --crop_factor --workers 0 --Nspots 20 --spotSize 2 --ModFac $modfac --PSFOTFscale $psf
    done
done

