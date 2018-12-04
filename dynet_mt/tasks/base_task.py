#!/usr/bin/env python3

from ..util import Logger, default_filename, set_default_arg
from ..models import add_model_type_args
from ..tokenizers import add_tokenizer_args
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
        raise NotImplementedError()

    def verify_args(self):
        pass

    def set_defaults(self, log):
        raise NotImplementedError()

    def get_logger(self):
        return Logger(verbose=self.args.verbose, out_file=self.args.log_file)

    def get_tokenizer(self):
        raise NotImplementedError()

    def get_data(self, tok, log=None):
        raise NotImplementedError()

    def run(self, data, dic_src, dic_tgt, tok):
        raise NotImplementedError()

    def main(self):
        raise NotImplementedError()


class TokenizerAndModelTask(BaseTask):
    """Tasks that require a model and a tokenizer to be specified"""
    desc = "Base for tasks requiring a tokenizer and a model"

    @staticmethod
    def parse_args(parser, task_subparser):
        # Get base args (mainly to retrieve --config-file)
        args = parse_args_and_yaml(parser, known_args_only=False)
        # Now parse with the task specific subparser and add to the
        # existing args
        args = parse_args_and_yaml(
            task_subparser,
            known_args_only=False,
            namespace=args
        )
        # Add model specific arguments
        add_model_type_args(args.model_type, task_subparser)
        # Add tokenizers specific arguments
        add_tokenizer_args(args.tokenizer_type, task_subparser)
        # Then parse again and add to the existing args
        # (and this time be strict about the arguments)
        final_args = parse_args_and_yaml(
            parser,
            known_args_only=True,
            namespace=args
        )
        print(final_args)
        return final_args

    def set_defaults(self, log):
        set_default_arg(
            self.args,
            key="tokenizer_file",
            default_value=default_filename(self.args, "train.cache.bin"),
            log=log
        )

    def get_logger(self):
        return Logger(verbose=self.args.verbose, out_file=self.args.log_file)

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
