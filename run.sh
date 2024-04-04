#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time=2:00:00
#SBATCH --clusters=htc
#SBATCH --job-name=vcl-diff
#SBATCH --gres=gpu:1 --constraint='gpu_mem:32GB'
#SBATCH --mem-per-cpu=16G
#SBATCH --output=reports/%j.out

module load Anaconda3/2023.09-0
module load CUDA/11.8.0

source activate $DATA/agd

python vcl.py