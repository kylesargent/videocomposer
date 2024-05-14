#!/bin/bash
# CUDA_VISIBLE_DEVICES=1

# Directory containing the paired image and video files
# input_directory="benchmarking/re10k"
# description="A real estate walkthrough video."
input_directory="benchmarking/co3d"
description="A 360-degree video of a common object."

# Loop over all png files in the directory
for image_file in "$input_directory"/*_input.png; do
    # Extract the base name without extension and directory (e.g., 00005 from 00005_input.png)
    base_name=$(basename "$image_file" "_input.png")
    
    # Construct the corresponding video filename
    video_file="${input_directory}/${base_name}_target.mp4"

    # Check if the video and image files exist
    if [ ! -f "$video_file" ]; then
        echo "Error: Video file for $base_name does not exist."
        exit 1
    fi    
    
    # Run the python command with the current image and video
    python run_net.py \
        --cfg configs/exp02_motion_transfer.yaml \
        --seed 9999 \
        --input_video "$video_file" \
        --image_path "$image_file" \
        --input_text_desc "$description" \
        --log_dir "${input_directory}/${base_name}"
done