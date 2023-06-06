#!/bin/bash

image_root="/Volumes/GoogleDrive/My Drive/01datasets/SIMRec/Lisa/September 2020"

shopt -s globstar

# for filename in "${image_root}"/**/*.tif; do
#     basefilename=$(basename "$filename")
#     if [[ $filename != *"488.tif"* ]]; then
#         continue
#     fi
#     if [ $basefilename == "SIM.tif" ] || [ $basefilename == "Widefield.tif" ]; then
#         continue
#     fi
#     dir=$(dirname "$filename")
#     echo $basefilename ---  $dir ---- $(dirname "$(dirname "${dir}")")
#     echo ""
# done



for filename in $(find '/Volumes/GoogleDrive/My Drive/01datasets/SIMRec/Lisa/September 2020' -name '*.tif' | xargs -0 process \;); do
    basefilename=$(basename "$filename")
    if [[ $filename != *"488.tif"* ]]; then
        continue
    fi
    echo $filename
done



