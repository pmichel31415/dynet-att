#!/bin/bash
source ~/.bashrc
NAME=medium_model_dropout_bidir_wembs_clip
# Run big model with attention and dropout
python train.py --dynet-mem 12000 --dynet-gpus 1 -v -de 512 -dh 512 --batch_size 32 --train_src en-de/train.en-de.low.filt.de --train_dst en-de/train.en-de.low.filt.en --valid_src en-de/valid.en-de.low.de --test_src en-de/test.en-de.low.de --valid_dst en-de/valid.en-de.low.en -en $NAME --test_every 5000 --check_valid_error_every 5000 --check_train_error_every 100 -ml 80 -dr 0.5 --num_epochs 20 --test_out results/${NAME}_test.de-en.en --learning_rate_decay 0.1 --beam_size 5 -att --train

echo "Done!"
