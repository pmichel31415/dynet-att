#!/usr/bin/env python3

from ..util import Logger, default_filename, set_default_arg
from ..command_line import print_config
from ..command_line import parse_args_and_yaml


class BaseTask(object):
    desc = "Base task class"

    def __init__(self, args):
        self.args = args
        # Set up logging
        self.log = self.get_logger()

    @staticmethod
    def add_args(parser):
        pass

    @staticmethod
    def parse_args(parser, task_subparser):
        return parse_args_and_yaml(parser)

    def verify_args(self):
        pass

    def set_defaults(self, log):
        set_default_arg(
            self.args,
            key="tokenizer_file",
            default_value=default_filename(self.args, "train.cache.bin"),
            log=log
        )

    def get_logger(self):
        return Logger(verbose=self.args.verbose, out_file=self.args.log_file)

    def get_tokenizer(self):
        raise NotImplementedError()

    def get_data(self, tok, log=None):
        raise NotImplementedError()

    def run(self, data, dic_src, dic_tgt, tok):
        raise NotImplementedError()

    def main(self):
        # Sanitize arguments and set defaults
        self.verify_args()
        self.set_defaults()
        # Print config
        print_config(self.args)
        # Get tokenizer (might need to train it if it's a subword model)
        tok = self.get_tokenizer()
        # Load, tokenize and numberize data
        dataset, dic_src, dic_tgt = self.get_data(tok)
        # RUN
        self.run(dataset, dic_src, dic_tgt, tok)
