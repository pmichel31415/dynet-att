#!/bin/bash
source ~/.bashrc
# Run big model with attention and dropout
python train.py --dynet-mem 10000 --dynet-gpus 1 -v -de 256 -dh 256 --batch_size 10 --train_src en-de/train.en-de.low.filt.de --train_dst en-de/train.en-de.low.filt.en --valid_src en-de/valid.en-de.low.de --test_src en-de/test.en-de.low.de --valid_dst en-de/valid.en-de.low.en -att -en smal_model_dropout --test_every 1000 --check_valid_error_every 1000 --check_train_error_every 100 -ml 80 -dr 0.5 --num_epochs 10

echo "Done!"
