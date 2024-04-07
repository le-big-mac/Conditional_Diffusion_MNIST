#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --clusters=htc
#SBATCH --job-name=vcl-diff
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=reports/%j.out

module load Anaconda3/2023.09-0
module load CUDA/11.8.0

source activate $DATA/agd

python restart_training.py "$@"