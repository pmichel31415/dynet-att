#!/usr/bin/env python3

from .base_tokenizer import BaseTokenizer


class SpaceTokenizer(BaseTokenizer):

    def _tokenize(self, string, lang):
        return string.split(" ")

    def _detokenize(self, tokens, lang):
        return " ".join(tokens)

    @staticmethod
    def load(filename):
        return SpaceTokenizer()

    @staticmethod
    def from_args(args):
        return SpaceTokenizer()
