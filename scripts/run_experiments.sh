#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=training_bert
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:40:00
#SBATCH --mem=12000M
#SBATCH --output=training_%A.out


source activate bert

srun python main.py --max_epochs 50 --freeze_lm --data_dir 'insert path to mutual here' --device 'cuda' --model_name 'bert'
