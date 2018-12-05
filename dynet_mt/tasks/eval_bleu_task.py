#!/usr/bin/env python3
import os.path


from ..util import Logger, default_filename, set_default_arg
from ..data import prepare_eval_batches, prepare_eval_data
from ..models import model_from_args
from ..tokenizers import tokenizer_from_args
from ..decoding import BeamSearch
from ..translator import Translator
from ..training import eval_bleu
from ..evaluation import bleu_to_str
from ..command_line import add_preprocessing_args
from ..command_line import add_model_args
from ..command_line import add_translation_args

from .base_task import TokenizerAndModelTask


class EvalBLEUTask(TokenizerAndModelTask):
    desc = "Evaluate BLEU score"

    @staticmethod
    def add_args(parser):
        add_preprocessing_args(parser)
        add_model_args(parser)
        add_translation_args(parser)
        # Training specfic arguments
        eval_group = parser.add_argument_group("Evaluation specific arguments")
        eval_group.add_argument("--eval-src", type=str,
                                help="Eval data in the source language")
        eval_group.add_argument("--eval-tgt", type=str,
                                help="Train data in the target language")
        eval_group.add_argument("--eval-batch-size", type=int, default=10)
        eval_group.add_argument("--max-tokens-per-eval-batch", type=int,
                                default=9999)

    def verify_args(self):
        for arg_name in ["eval_src", "eval_tgt"]:
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
        return prepare_eval_data(self.args, tok, log=self.log)

    def run(self, data, dic_src, dic_tgt, tok):
        # Prepare batches
        eval_batches = prepare_eval_batches(
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
        bleu = eval_bleu(
            translator,
            eval_batches,
            data["eval_src_words"],
            data["eval_tgt_words"],
            log=self.log,
        )
        # Print
        print(bleu_to_str(bleu))