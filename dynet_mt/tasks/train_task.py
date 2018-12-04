#!/usr/bin/env python3
import os.path

from dynn.data.caching import cached_to_file

from ..util import Logger, default_filename, set_default_arg
from ..data import prepare_training_batches, prepare_training_data
from ..models import model_from_args
from ..tokenizers import tokenizer_from_args
from ..training import train
from ..optimizers import optimizer_from_args
from ..objectives import objective_from_args
from ..command_line import add_preprocessing_args
from ..command_line import add_optimization_args
from ..command_line import add_model_args

from .base_task import TokenizerAndModelTask


class TrainTask(TokenizerAndModelTask):
    desc = "Train an MT model"

    @staticmethod
    def add_args(parser):
        add_preprocessing_args(parser)
        add_model_args(parser)
        add_optimization_args(parser)
        # Training specfic arguments
        train_group = parser.add_argument_group("Training specific arguments")
        train_group.add_argument("--train-src", type=str,
                                 help="Train data in the source language")
        train_group.add_argument("--train-tgt", type=str,
                                 help="Train data in the target language")
        train_group.add_argument("--valid-src", type=str,
                                 help="Validation data in the source language")
        train_group.add_argument("--valid-tgt", type=str,
                                 help="Validation data in the target language")
        train_group.add_argument("--train-data-cache", type=str,
                                 help="This file will be used to cache the "
                                 "preprocessed training data. If not"
                                 " specified, it will default to "
                                 "[output_dir]/[exp_name].train.cache.bin")
        train_group.add_argument("--clear-train-data-cache",
                                 action="store_true",
                                 help="Preprocess the train data again "
                                 "(and update \"--train-data-cache\")")

    def verify_args(self):
        for arg_name in ["train_src", "train_tgt", "valid_src", "valid_tgt"]:
            arg_flag = arg_name.replace("_", "-")
            arg_val = getattr(self.args, arg_name, None)
            if arg_val is None:
                raise ValueError(f"\"--{arg_flag}\" not specified.")
            else:
                filename = os.path.abspath(arg_val)
                if not os.path.isfile(filename):
                    raise ValueError(
                        f" Given \"--{arg_flag}\" doesn't exist ({filename})"
                    )

    def set_defaults(self):
        set_default_arg(
            self.args,
            key="train_data_cache",
            default_value=default_filename(self.args, "train.cache.bin"),
            log=self.log
        )
        set_default_arg(
            self.args,
            key="model_save_file",
            default_value=default_filename(self.args, "model.npz"),
            log=self.log
        )
        set_default_arg(
            self.args,
            key="model_load_file",
            default_value=default_filename(self.args, "model.npz"),
            log=self.log
        )
        set_default_arg(
            self.args,
            key="dic_src",
            default_value=default_filename(self.args, "dic.src"),
            log=self.log
        )
        set_default_arg(
            self.args,
            key="dic_tgt",
            default_value=default_filename(self.args, "dic.tgt"),
            log=self.log
        )

    def get_logger(self):
        return Logger(verbose=self.args.verbose, out_file=self.args.log_file)

    def get_tokenizer(self):
        return tokenizer_from_args(self.args)

    def get_data(self, tok):
        caching_func = cached_to_file(self.args.train_data_cache)
        prepare_func = caching_func(prepare_training_data)
        return prepare_func(
            self.args,
            tok,
            update_cache=self.args.clear_train_data_cache,
            log=self.log
        )

    def run(self, data, dic_src, dic_tgt, tok):
        # Prepare batches
        train_batches, valid_batches = prepare_training_batches(
            self.args,
            data,
            dic_src,
            dic_tgt
        )
        # Create model
        model = model_from_args(self.args, dic_src, dic_tgt)
        # Setup optimizer
        optimizer = optimizer_from_args(self.args, model.pc)
        # Setup objective
        objective = objective_from_args(self.args)
        # Train
        train(
            self.args,
            model,
            objective,
            optimizer,
            train_batches,
            valid_batches,
            log=self.log
        )
