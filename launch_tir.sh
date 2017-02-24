#!/bin/bash

# Bare
sbatch --gres=gpu:1 --mem=15000 -J medium_model_softmax -o log_medium_model_softmax.txt --time=2-12 run_medium_model_dropout_softmax.sh

# Bidir
sbatch --gres=gpu:1 --mem=15000 -J medium_model_softmax -o log_medium_model_softmax.txt --time=2-12 run_medium_model_dropout_softmax.sh

# Bidir + word embs
sbatch --gres=gpu:1 --mem=15000 -J medium_model_softmax -o log_medium_model_softmax.txt --time=2-12 run_medium_model_dropout_softmax.sh
