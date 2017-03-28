#!/bin/bash
source ~/.bashrc
NAME=big_model_dropout_bidir
# Run big model with attention and dropout
python train.py --dynet-mem 10000 --dynet-gpus 1 -v -de 1000 -dh 1000 --batch_size 5 --train_src en-de/train.en-de.low.filt.de --train_dst en-de/train.en-de.low.filt.en --valid_src en-de/valid.en-de.low.de --test_src en-de/test.en-de.low.de --valid_dst en-de/valid.en-de.low.en -en $NAME --test_every 1000 --check_valid_error_every 1000 --check_train_error_every 100 -ml 80 -dr 0.5 --num_epochs 10 --test_out results/${NAME}_test_final.de-en.en --beam_size 5 -att -bid --test -m ${NAME}_model.txt

echo "Done!"
