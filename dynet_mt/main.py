#!/usr/bin/env python3

from .command_line import parse_and_get_args
from .training import train, sanitize_training_args
from .data import prepare_training_data, prepare_training_batches
from .models import model_from_args
from .optimizers import optimizer_from_args
from .objectives import objective_from_args
from .tokenizers import tokenizer_from_args
from .util import Logger


def main():
    args = parse_and_get_args()
    if args.task == "train":
        # Set up logging
        log = Logger(verbose=args.verbose, out_file=args.log_file)
        # Sanitize arguments
        args = sanitize_training_args(args)
        # Get tokenizer (might need to train it if it's a subword model)
        tok = tokenizer_from_args(args)
        # Load, tokenize and numberize data
        dataset, dic_src, dic_tgt = prepare_training_data(args, tok, log=log)
        # Prepare batches
        train_batches, valid_batches = prepare_training_batches(
            args,
            dataset,
            dic_src,
            dic_tgt
        )
        # Create model
        model = model_from_args(args, dic_src, dic_tgt)
        # Setup optimizer
        optimizer = optimizer_from_args(args, model.pc)
        # Setup objective
        objective = objective_from_args(args)
        # Train
        train(
            args,
            model,
            objective,
            optimizer,
            train_batches,
            valid_batches,
            log=log
        )


if __name__ == "__main__":
    main()
