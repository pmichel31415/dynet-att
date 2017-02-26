#!/bin/bash
source ~/.bashrc
NAME=medium_model_dropout_softmax
# Run training
python train.py --dynet-mem 12000 --dynet-gpus 1 -v -de 512 -dh 512 --batch_size 32 --train_src en-de/train.en-de.low.filt.de --train_dst en-de/train.en-de.low.filt.en --valid_src en-de/valid.en-de.low.de --test_src en-de/test.en-de.low.de --valid_dst en-de/valid.en-de.low.en -en $NAME --test_every 5000 --check_valid_error_every 5000 --check_train_error_every 100 -ml 80 -dr 0.2 --num_epochs 30 --test_out results/${NAME}_test.de-en.en --learning_rate_decay 0.01 --beam_size 5 --train

echo "Done training!"

# Run testing
for i in 1 2 3 4 5;
    do
    python train.py --dynet-mem 12000 --dynet-gpus 1 -v -de 512 -dh 512 --batch_size 32 --train_src en-de/train.en-de.low.filt.de --train_dst en-de/train.en-de.low.filt.en --valid_src en-de/valid.en-de.low.de --test_src en-de/test.en-de.low.de --valid_dst en-de/valid.en-de.low.en -en ${NAME}_beam_$i --test_every 5000 --check_valid_error_every 5000 --check_train_error_every 100 -ml 80 -dr 0.2 --num_epochs 50 --test_out results/${NAME}_beam_${i}_test.de-en.en --beam_size $i --test -m ${NAME}_model.txt
done
echo "Done testing!"
