#!/bin/bash
#SBATCH -J hawaiian_gpt2
#SBATCH -o logs/gpt2.o%j
#SBATCH -e logs/gpt2.e%j
#SBATCH -p gpu-shared
#SBATCH -A TG-MCB180035
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 48:00:00

module purge
module load gpu
module load singularitypro


python3.11 train_llm.py

# Run batch job within hawaiian_llm_scripts folder: sbatch train_exec.sh
# To check status:  squeue -u $USER
# To cancel: scancel <job number>
