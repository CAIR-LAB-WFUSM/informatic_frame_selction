#!/bin/bash
#SBATCH --job-name=single_video_frame_evaluate
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --time=6-06:00:00   # specify job time
#SBATCH --mem=32G         # specify memory
#SBATCH --output=%x_%j.log

module load cuda-toolkit/12.2.2  # load necessary modules

srun python single_video_frame_evaluate.py  # run your job