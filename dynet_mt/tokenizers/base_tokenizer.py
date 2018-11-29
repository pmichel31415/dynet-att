#!/usr/bin/env python3


class BaseTokenizer(object):
    """Base tokenizer class"""

    def tokenize(self, str_or_list, lang="en"):
        if isinstance(str_or_list, str):
            return self._tokenize(str_or_list, lang=lang)
        elif isinstance(str_or_list, list):
            return [self.tokenize(elem, lang=lang) for elem in str_or_list]
        else:
            raise ValueError(
                f"{str_or_list.__class__.__name__} is neither a str or a list"
            )

    def detokenize(self, list_or_list_of_lists, lang="en"):
        if isinstance(list_or_list_of_lists, list):
            if isinstance(list_or_list_of_lists[0], str):
                return self._detokenize(list_or_list_of_lists, lang=lang)
            else:
                return [
                    self.detokenize(elem, lang=lang)
                    for elem in list_or_list_of_lists
                ]
        else:
            raise ValueError(
                f"{list_or_list_of_lists.__class__.__name__} not a list"
            )

    def _tokenize(self, string, lang):
        raise NotImplementedError()

    def _detokenize(self, tokens, lang):
        raise NotImplementedError()

    def save(self, filename):
        pass

    @staticmethod
    def load(filename):
        raise NotImplementedError()

    @staticmethod
    def from_args(args):
        raise NotImplementedError()

    @staticmethod
    def add_args(parser):
        pass
