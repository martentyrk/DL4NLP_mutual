#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=training_bert
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm_%A.out


source activate bert

srun python main.py --max_epochs 50 --data_dir '/home/scur0667/DL4NLP_mutual/data/mutual' --device 'gpu'