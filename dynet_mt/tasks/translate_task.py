#!/usr/bin/env python3
import os.path


from ..util import Logger, default_filename, set_default_arg
from ..data import prepare_translate_batches, prepare_translate_data
from ..models import model_from_args, add_model_type_args
from ..tokenizers import tokenizer_from_args, add_tokenizer_args
from ..decoding import BeamSearch
from ..translator import Translator
from ..training import translate
from ..command_line import add_preprocessing_args
from ..command_line import add_model_args
from ..command_line import add_translation_args
from ..command_line import parse_args_and_yaml

from .base_task import BaseTask


class TranslateTask(BaseTask):
    desc = "Train an MT model"

    @staticmethod
    def add_args(parser):
        add_preprocessing_args(parser)
        add_model_args(parser)
        add_translation_args(parser)
        # Training specfic arguments
        eval_group = parser.add_argument_group("Evaluation specific arguments")
        eval_group.add_argument("--trans-src", type=str,
                                help="Source text to translate")
        eval_group.add_argument("--out-file", type=str,
                                help="Output file (defaults to stdout)")
        eval_group.add_argument("--trans-batch-size", type=int, default=1)
        eval_group.add_argument("--max-tokens-per-trans-batch", type=int,
                                default=9999)

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
        return final_args

    def verify_args(self):
        for arg_name in ["trans_src"]:
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
        self.args.pretrained_model = True

    def get_logger(self):
        return Logger(verbose=self.args.verbose, out_file=self.args.log_file)

    def get_tokenizer(self):
        return tokenizer_from_args(self.args)

    def get_data(self, tok):
        return prepare_translate_data(self.args, tok, log=self.log)

    def run(self, data, dic_src, dic_tgt, tok):
        # Prepare batches
        trans_batches = prepare_translate_batches(
            self.args,
            data,
            dic_src,
            dic_tgt
        )
        # Create model
        model = model_from_args(self.args, dic_src, dic_tgt)
        # Translator
        translator = Translator(
            model,
            BeamSearch(self.args.beam_size, self.args.lenpen),
            tok,
            replace_unk=self.args.replace_unk,
        )
        # Train
        hyps = translate(
            translator,
            trans_batches,
            data["trans_src_words"],
            log=self.log,
        )
        # Print
        log_trans = Logger(out_file=self.args.out_file)
        for hyp in hyps:
            log_trans(hyp)
