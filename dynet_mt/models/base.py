#!/usr/bin/env python3
import dynet as dy
import dynn


class BaseSeq2Seq(dynn.layers.ParametrizedLayer):
    """Base seq2seq model"""

    def __init__(self, dic_src, dic_tgt):
        pc = dy.ParameterCollection()
        super(BaseSeq2Seq, self).__init__(pc, "seq2seq")
        self.dic_src = dic_src
        self.dic_tgt = dic_tgt

    def encode(self, src):
        raise NotImplementedError()

    def embed_word(self, word):
        raise NotImplementedError()

    @property
    def sos(self):
        raise NotImplementedError()

    @property
    def initial_decoder_state(self):
        raise NotImplementedError()

    def __call__(self, src, tgt):
        return self.logits(src, tgt)

    def logits(self, src, tgt):
        raise NotImplementedError()
