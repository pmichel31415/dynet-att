#!/usr/bin/env python3

import sacremoses
from dynn import io

from .base_tokenizer import BaseTokenizer


class MosesTokenizer(BaseTokenizer):

    def __init__(self, escape=True):
        self.escape = escape
        self.toks = {}
        self.detoks = {}

    def lang_tok(self, lang):
        if lang not in self.toks:
            self.toks[lang] = sacremoses.MosesTokenizer(lang=lang)
        return self.toks[lang]

    def lang_detok(self, lang):
        if lang not in self.detoks:
            self.detoks[lang] = sacremoses.MosesDetokenizer(lang=lang)
        return self.detoks[lang]

    def _tokenize(self, string, lang):
        return self.lang_tok(lang).tokenize(string, escape=self.escape)

    def _detokenize(self, tokens, lang):
        return self.lang_detok(lang).detokenize(tokens, unescape=self.escape)

    def save(self, filename):
        io.savetxt(
            filename,
            [MosesTokenizer.__class__.__name__, str(self.escape)]
        )

    @staticmethod
    def load(filename):
        lines = io.loadtxt(filename)
        if len(lines) != 2 or lines[0] != MosesTokenizer.__class__.__name__:
            raise ValueError(
                f"{filename} has incorrect format for MosesTokenizer"
            )
        return MosesTokenizer(escape=bool(lines[1]))

    @staticmethod
    def from_args(args):
        return MosesTokenizer(escape=args.moses_escape)

    @staticmethod
    def add_args(parser):
        moses_group = parser.add_argument_group("Moses tokenizer")
        moses_group.add_argument("--moses-escape", action="store_true")
