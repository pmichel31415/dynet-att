#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -t 0
source ~/.bashrc
EN=best_unigram
CFG_FILE=temp/`uuidgen`.yaml
cp config/best_config.yaml $CFG_FILE
# Training
python run.py -c $CFG_FILE -e train --exp_name $EN > output/log_${EN}.txt 2>&1
# Testing
python run.py -c $CFG_FILE -e test --exp_name $EN >> output/log_${EN}.txt 2>&1
