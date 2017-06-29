from __future__ import division, print_function

import numpy as np
import dynet as dy

class LanguageModel(object):
    def p_next(self, sent):
        pass

    def init(self):
        pass
    
    def p_next_expr(self, sent):
        return dy.inputTensor(self.p_next(sent))

    def fit(self, corpus):
        pass

    def save(self, filename):
        pass

    def load(self, filename):
        pass


class UniformLanguageModel(LanguageModel):
    def __init__(self, w2id):
        self.n = len(id2w)
    
    def p_next(self, sent):
        return np.ones(self.n) / self.n


class UnigramLanguageModel(LanguageModel):
    def __init__(self, w2id, eps=0):
        self.id2w = w2id
        self.eps = eps
        self.unigrams = np.ones(len(self.id2w)) / len(self.id2w)

    def init(self):
        self.u_e = dy.inputTensor(self.unigrams)

    def p_next(self, sent):
        return self.unigrams

    def p_next_expr(self, sent):
        return self.u_e

    def fit(self, corpus):
        self.unigrams = np.zeros(len(self.id2w)) + self.eps
        for sent in corpus:
            for w in sent:
                self.unigrams[w] += 1
        self.unigrams /= self.unigrams.sum()

    def save(self, filename):
        np.save(filename, self.unigrams)
    
    def load(self, filename):
        self.unigrams = np.load(filename)
