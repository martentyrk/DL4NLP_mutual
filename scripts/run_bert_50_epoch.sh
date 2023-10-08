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

srun python main.py --max_epochs 50 --freeze_lm --lr_scheduler --data_dir '/home/scur0670/DL4NLP_mutual/data/mutual' --device 'cuda' --model_name 'bert'
