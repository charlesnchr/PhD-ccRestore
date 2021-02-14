#!/bin/bash

model_dir="/Volumes/GoogleDrive/My Drive/0main/MLSIM-Revision-2020/mlsim-runs"

# test=$("${model_dir}"/**/*.pth tr "" "\n")

for filename in "${model_dir}"/**/final.pth; do
    echo $filename
done



