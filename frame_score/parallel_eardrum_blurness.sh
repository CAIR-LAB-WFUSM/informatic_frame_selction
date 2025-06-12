#!/bin/bash
#SBATCH --job-name=blurness_parallel
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:0
#SBATCH --time=6-23:00:00   # specify job time
#SBATCH --mem=16G         # specify memory
#SBATCH --output=parallel_eardrum_%j_%x_%u.log

module load cuda-toolkit/12.2.2  # load necessary modules

srun python parallel_eardrum_blurness.py  # run your job
