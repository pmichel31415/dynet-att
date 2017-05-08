#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -t 0
#SBATCH -o output/log_de-en_best.txt
source ~/.bashrc
# Training
python run.py -c config/best_config.yaml -e train
echo "Done training!"

# Testing
python run.py -c config/best_config.yaml -e test
echo "Done testing!"
