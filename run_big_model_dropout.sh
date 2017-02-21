#!/bin/bash

# Run big model with attention and dropout
python train.py --dynet-mem 20000 -v -de 1000 -dh 1000 --batch_size 5 --train_src en-de/train.en-de.low.filt.de --train_dst en-de/train.en-de.low.filt.en --valid_src en-de/valid.en-de.low.de --test_src en-de/test.en-de.low.de --valid_dst en-de/valid.en-de.low.en -att -en big_model_dropout --test_every 1000 --check_valid_error_every 1000 --check_train_error_every 100 -ml 80 -dr 0.5 &> log_big_model_dropout.txt
