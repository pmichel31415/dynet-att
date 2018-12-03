#!/usr/bin/env python3

from dynn.data.batching import SequenceBatch


class Translator(object):
    """Handles translation end to end.

    Args:
        model (models.BaseSeq2Seq): Neural model
        decoding (decoding.Decoding): Decoding strategy
        tokenizer (tokenizers.Tokenizer): [De]tokenizer
        lexicon (dict, optional): Bilingual lexicon mapping source words to
            target words
    """

    def __init__(
        self,
        model,
        decoding,
        tokenizer,
        lexicon=None,
    ):

        self.model = model
        self.decoding = decoding
        self.tok = tokenizer
        self.lex = lexicon

    def __call__(self, src, src_words=None):
        # Preprocess input
        batch, src_words_ = self.input_to_batch(src)
        if src_words is None:
            src_words = src_words_
        # Run the model
        hyp_ids, aligns = self.decoding(self.model, batch)
        # Convert ids to words
        hyp_words = [self.model.dic_tgt.string(hyp) for hyp in hyp_ids]
        # Unk replacement
        if src_words is not None and aligns is not None:
            for b, hyp in enumerate(hyp_words):
                hyp_words[b] = self.unk_replace(src_words[b], hyp, aligns[b])
        # Detokenize
        hyp_words = [self.tok.detokenize(x) for x in hyp_words]
        # Return translations (either as a list or as a string)
        str_input = isinstance(src, str)
        id_list_input = isinstance(src, list) and isinstance(src[0], int)
        if str_input or id_list_input:
            return hyp_words[0]
        else:
            return hyp_words

    def input_to_batch(self, src):
        """Handle the input, returns a SequenceBatch and a list of lists of
        words (possibly None)"""
        if isinstance(src, list):
            # A list can be several things
            if isinstance(src[0], int):
                # Either a list of ints (numberized sentence)
                src_words = self.model.dic_src.string(src, join_with=None)
                return SequenceBatch([src]), [src_words]
            elif isinstance(src[0], list):
                # Otherwise it's either a list of strings or
                # a list of list of ids
                all_src_words = [self.tokenize_maybe(x) for x in src]
                if isinstance(all_src_words[0][0], int):
                    # Batch of numberized sentences
                    src_words = [self.model.dic_src.string(x, join_with=None)
                                 for x in all_src_words]
                    return SequenceBatch(all_src_words), src_words
                elif isinstance(all_src_words[0][0], str):
                    # Batch of sentences, numberize them
                    all_src_ids = self.model.dic_src.numberize(all_src_words)
                    return SequenceBatch(all_src_ids), all_src_words
                else:
                    raise ValueError(
                        f"Invalid input in Translator: list of "
                        f"{src[0].__class__.__name__}. Should be either a list"
                        " of strings, or a list of numberized sentences "
                        "(lists of int)"
                    )
            else:
                raise ValueError(
                    "Invalid input in Translator: list of "
                    f"{src[0].__class__.__name__}. Should be either a list of "
                    "strings, or a list of numberized sentences (lists of int)"
                )
        elif isinstance(src, str):
            # If the input is a string
            src_words = self.tok.tokenize(src)
            src_ids = self.model.dic_src.numberize(src_words)
            return SequenceBatch(src_ids), src_words
        elif isinstance(src, SequenceBatch):
            return src, None
        else:
            raise ValueError(
                f"Invalid input in Translator: {src.__class__.__name__}. "
                "Should be either a string, a list or a "
                "dynn.data.batching.SequenceBatch"
            )

    def tokenize_maybe(self, maybe_string):
        if isinstance(maybe_string, str):
            return self.tok.tokenize(maybe_string)
        else:
            return maybe_string

    def unk_replace(self, src_words, hyp_words, align):
        """Replace unks according to alignements"""
        if src_words is None or align is None:
            return hyp_words
        #
        for i, w in enumerate(hyp_words):
            if w == self.model.dic_tgt.unk_tok:
                src_word = src_words[align[i]]
                if self.lex is None:
                    hyp_words[i] = src_word
                else:
                    hyp_words[i] = self.lex[src_word]
        return hyp_words
