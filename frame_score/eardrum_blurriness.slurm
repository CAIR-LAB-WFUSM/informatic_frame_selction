#!/bin/bash
#SBATCH --job-name=haolu_job
#SBATCH --partition=ciaq
#SBATCH --gres=gpu:1
#SBATCH --time=6-06:00:00   # specify job time
#SBATCH --mem=16G         # specify memory
#SBATCH --output=eardrum_blurness_%j_%x_%u.log

module load cuda-toolkit/12.2.2  # load necessary modules

srun python eardrum_blurness.py  # run your job
