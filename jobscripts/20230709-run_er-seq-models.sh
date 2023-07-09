#!/bin/bash
: ' ----------------------------------------
* Creation Time : Sun Jul  9 19:39:11 2023
* Author : Charles N. Christensen
* Github : github.com/charlesnchr
----------------------------------------'

new_root_path="/Users/cc/Desktop/seq-trainingdata/20201213_partitioned_256"
new_root_validation_path="/Users/cc/Desktop/seq-trainingdata/20201213_partitioned_256"
base_output_folder="/Users/cc/Desktop"

# Loop over each model directory
for dir in /Users/cc/Desktop/ER-seq_/*/
do
    # skip everything but 20210216-ER-seq-nch_in3
    # if [[ "$dir" != *"20210216-ER-seq-nch_in3"* ]]; then
    #     continue
    # fi

    # Remove trailing slash
    dir=${dir%/}

    # Extract model name from the directory path
    model=$(basename "$dir")

    # Define new output path for the model
    new_out_path="$base_output_folder/$model"

    # Extract parameters from the log.txt file
    args=$(awk -F: '/ARGS: run.py/{gsub(/ARGS: run.py --/, ""); print}' "$dir/log.txt")

    # Remove old --root and --rootValidation parameters
    args=$(echo $args | sed -E 's/--root [^ ]+|--rootValidation [^ ]+//g')

    # Add new --root and --rootValidation parameters
    args="$args --root $new_root_path --rootValidation $new_root_validation_path --out $new_out_path"

    # Add weights file to the parameters
    args="--$args --weights $dir/final.pth --test --mps --disable_wandb --nplot 0 --ntest 2000 --batchSize_test 10"

    # echo args with newline
    echo "$args\n"

    # Run your script with the extracted parameters
    python run.py $args


    echo "Finished processing $model"
done
