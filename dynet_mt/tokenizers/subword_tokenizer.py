#!/usr/bin/env python3
import sentencepiece as sp
from dynn import io

from ..util import default_filename
from .base_tokenizer import BaseTokenizer


class SubwordTokenizer(BaseTokenizer):

    def __init__(self, prefix):
        self.prefix = prefix
        self.spp = sp.SentencePieceProcessor()
        self.spp.Load(f"{self.prefix}.model")

    def _tokenize(self, string, lang):
        return self.spp.EncodeAsPieces(string)

    def _detokenize(self, tokens, lang):
        return "".join(tokens).replace("▁", " ")[1:]

    def save(self, filename):
        io.savetxt(
            filename,
            [SubwordTokenizer.__class__.__name__, self.prefix]
        )

    @staticmethod
    def load(filename):
        lines = io.loadtxt(filename)
        if len(lines) != 2 or lines[0] != SubwordTokenizer.__class__.__name__:
            raise ValueError(
                f"{filename} has incorrect format for SubwordTokenizer"
            )
        return SubwordTokenizer(prefix=lines[1])

    @staticmethod
    def from_args(args):
        # Default model prefix
        if args.subword_model_prefix is None:
            prefix = f"{args.subword_algo}.{args.subword_voc_size}"
            args.subword_model_prefix = default_filename(args, prefix)
        # Otherewise learn the subword model on the fly
        if len(args.subword_train_files) == 0:
            raise ValueError(
                "You need to ether provide the prefix to an existing "
                "subword model via \"--subword-model-prefix\" or "
                "training files to train the subword model via "
                "\"--subword-train-files\""
            )
        train_files_list = ",".join(args.subword_train_files)
        arguments = (
            f"--input={train_files_list} "
            f"--model_prefix={args.subword_model_prefix} "
            f"--model_type={args.subword_algo} "
            f"--vocab_size={args.subword_voc_size}"
        )
        sp.SentencePieceTrainer.Train(arguments)
        return SubwordTokenizer(args.subword_model_prefix)

    @staticmethod
    def add_args(parser):
        subword_group = parser.add_argument_group("Subword tokenizer")
        subword_group.add_argument("--subword-model-prefix", type=str)
        subword_group.add_argument("--subword-train-files", type=str,
                                   nargs="*", default=[])
        subword_group.add_argument("--subword-voc-size", type=int)
        subword_group.add_argument("--subword-algo", type=str,
                                   choices=["bpe", "unigram"])
