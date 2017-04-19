#!/bin/bash
source ~/.bashrc
NAME=medium_model_dropout_bidir_sgd
# Run big model with attention and dropout
python run.py --dynet-seed 3416888402 --dynet-gpus 1 --dynet-mem 512 -c config/best_config.yaml -e train

echo "Done training!"

# Run big model with attention and dropout
python run.py --dynet-seed 3416888402 --dynet-gpus 1 --dynet-mem 512 -c config/best_config.yaml -e test
echo "Done testing!"
