#!/bin/bash
source ~/.bashrc
NAME=medium_model_dropout_bidir_sgd
# Run big model with attention and dropout
python run.py -c config/best_config.yaml -e train

echo "Done training!"

# Run big model with attention and dropout
python run.py -c config/best_config.yaml -e test
echo "Done testing!"
