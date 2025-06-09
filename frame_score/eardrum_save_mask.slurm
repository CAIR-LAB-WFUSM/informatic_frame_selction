#!/bin/bash
#SBATCH --job-name=haolu_job
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:4
#SBATCH --time=6-23:00:00   # specify job time
#SBATCH --mem=16G         # specify memory
#SBATCH --output=eardrum_save_mask_%j_%x_%u.log

module load cuda-toolkit/12.2.2  # load necessary modules

srun python eardrum_save_mask.py  # run your job
