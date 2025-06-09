#!/bin/bash
#SBATCH --job-name=eardrum_mask_and_area
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:4
#SBATCH --time=6-23:00:00   # specify job time
#SBATCH --mem=16G         # specify memory
#SBATCH --output=eardrum_mask_and_area%j_%x_%u.log

module load cuda-toolkit/12.2.2  # load necessary modules

srun python eardrum_mask_and_area.py  # run your job
