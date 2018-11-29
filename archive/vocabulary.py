#!/usr/bin/env python3

import pickle
import numpy as np
from collections import defaultdict


def zero():
    return 0


def dd():
    return defaultdict(zero)


def read_literal_lexicon(filename):
    """Load translation lexicon between words (no UNKs)"""
    lex = dict()
    with open(filename, "r") as f:
        for l in f:
            s, t, p = l.split()
            p = float(p)
            if s not in lex:
                lex[s] = [[], []]
            lex[s][0].append(t)
            lex[s][1].append(p)
    return lex


def reverse_dic(dic):
    """Get the inverse mapping from an injective dictionary

    dic[key] = value <==> reverse_dic(dic)[value] = key

    Args:
        dic (defaultdict): Dictionary to reverse

    Returns:
        defaultdict: Reversed dictionary
    """
    rev_dic = dict()
    for k, v in dic.items():
        rev_dic[v] = k
    return rev_dic


def read_dic(file, max_size=200000, min_freq=-1):
    """Read dictionary from corpus

    Args:
        file (str): [description]

    Keyword Arguments:
        max_size (int): Only the top max_size words (by frequency) are stored,
            the rest is UNKed (default: {20000})
        min_freq (int): Disregard words with frequency <= min_freq (default: {1})

    Returns:
        Dictionary
        defaultdict
    """
    dic = dict()
    freqs = defaultdict(zero)
    dic["UNK"], dic["SOS"], dic["EOS"] = 0, 1, 2
    with open(file, "r") as f:
        for l in f:
            sent = l.strip().split()
            for word in sent:
                freqs[word] += 1
    sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    for i in range(min(max_size, len(sorted_words))):
        word, freq = sorted_words[i]
        if freq < min_freq:
            continue
        dic[word] = len(dic)

    return dic, reverse_dic(dic)


def read_lexicon(filename, dic_src, dic_trg, k=3):
    lex = defaultdict(dd)
    with open(filename, "r") as f:
        for l in f:
            source_word, target_word, prob = l.split()
            s_i = dic_src[source_word] if source_word in dic_src else dic_src["UNK"]
            t_i = dic_trg[target_word] if target_word in dic_trg else dic_trg["UNK"]
            lex[s_i][t_i] += float(prob)
    # Only retain top-k translations
    for s_w, v in lex.items():
        top_k = sorted(v.items(), key=lambda x: x[1])[-k:]
        lex[s_w] = dict(top_k)
    # Normalize the whole thing
    for s_w, v in lex.items():
        s = sum(v.values())
        for t_w, p in v.items():
            v[t_w] = p / s
    # to list of list (more efficient?)
    list_lex = [[[], []]] * len(dic_src)
    for s_w, v in lex.items():
        list_lex[s_w] = [[], []]
        for t_w, p in v.items():
            list_lex[s_w][0].append(t_w)
            list_lex[s_w][1].append(p)
    return list_lex


class Lexicon(object):
    """The lexicon class holds a lot of external information about the data.

    - dictionary (word -> ID and vice versa)
    - translation lexicons
    - Conditional lengths probabilities

    """

    def __init__(self):
        self.dic_src = None   # Word to id (source)
        self.dic_tgt = None   # Word to id (target)
        self.id2ws = None   # Id to word (source)
        self.id2wt = None   # Id to word (target)
        self.ws2t = None    # Source to target (words)
        self.wt2s = None    # Target to source (words)
        self.ids2t = None   # Source to target (ids)
        self.idt2s = None   # Target to source (ids)
        self.kbest = None   # K-best list of synonyms
        # np.zeros((500, 500))  # Conditional probabilities of lengths
        self.p_L = defaultdict(lambda: 0)

    def sents_to_ids(self, sents, trg=False, add_pad=True):
        return [self.sent_to_ids(s, trg=trg, add_pad=add_pad) for s in sents]

    def sent_to_ids(self, sent, trg=False, add_pad=True):
        dic = self.w2idt if trg else self.w2ids
        x = [dic[w] if w in dic else dic["UNK"] for w in sent]
        if add_pad:
            x.insert(0, dic["SOS"])
            x.append(dic["EOS"])
        return x

    def ids_to_sent(self, x, trg=False, cut_pad=True):
        dic = self.id2wt if trg else self.id2ws
        sent = [dic[i] for i in x]
        if cut_pad:
            sent = sent[1:-1]
        return sent

    def translate(self, word, reverse=False):
        dic = self.wt2s if reverse else self.ws2t
        if word not in dic:
            return word
        w, p = max(zip(*dic[word]), key=lambda x: x[1])
        if p <= 0.5:
            d = np.asarray(dic[word][1])
            d = d / d.sum()
            return np.random.choice(dic[word][0], p=d)
        else:
            return w

    def init(self, opt):
        # Load literal lexicon
        if opt.lex_s2t is not None:
            self.ws2t = read_literal_lexicon(opt.lex_s2t)
        if opt.lex_t2s is not None:
            self.wt2s = read_literal_lexicon(opt.lex_t2s)
        # Read dictionaries from training files
        self.w2ids, self.id2ws = read_dic(opt.train_src,
                                          max_size=opt.src_vocab_size,
                                          min_freq=opt.min_freq)
        self.w2idt, self.id2wt = read_dic(opt.train_trg,
                                          max_size=opt.trg_vocab_size,
                                          min_freq=opt.min_freq)

        # Read lexicons
        if opt.lex_s2t is not None:
            self.ids2t = read_lexicon(opt.lex_s2t, self.w2ids, self.w2idt)
        if opt.lex_t2s is not None:
            self.idt2s = read_lexicon(opt.lex_t2s, self.w2idt, self.w2ids)
        # Learn length translation probabilities
        self.learn_length_stats(opt)
        # Load synonyms
        self.load_kbest_synonyms(opt.synonyms_file, k=opt.kbest_synonyms)

    def learn_length_stats(self, opt):
        max_len = 200 + 2
        self.p_L = np.zeros((max_len, max_len))
        with open(opt.train_src, "r") as f_src:
            with open(opt.train_trg, "r") as f_trg:
                for sent_s, sent_t in zip(f_src, f_trg):
                    ls, lt = len(sent_s.strip().split()), len(
                        sent_t.strip().split())
                    if ls <= max_len and lt <= max_len:
                        self.p_L[ls + 2, lt + 2] += 1
        self.p_L += 0.0  # Smoothing a little bit
        # Normalize
        self.p_L /= self.p_L.sum(axis=-1).reshape((max_len, 1))+1e-20

    def compute_unigrams(self, corpus, lang="src", laplace_smoothing=0):
        voc_size = len(self.id2ws) if lang == "src" else len(self.id2wt)
        unigrams = np.zeros(voc_size) + laplace_smoothing
        for sent in corpus:
            for w_id in sent[1:]:
                unigrams[w_id] += 1
        unigrams /= unigrams.sum()
        return unigrams

    def estimate_unigrams(self, corpus):
        assert self.ids2t is not None, "A lexicon is needed to estimate unigrams in the target language given the source"
        src_unigrams = self.compute_unigrams(corpus)
        trg_unigrams = np.zeros(len(self.id2wt))
        for x, p_x in enumerate(src_unigrams):
            ys, p_ys_x = self.ids2t[x]
            trg_unigrams[ys] += [p_y_x * p_x for p_y_x in p_ys_x]
        return trg_unigrams

    def load_kbest_synonyms(self, filename, k=0):
        if k < 1:
            return
        self.kbest = [[] for i in self.id2wt.keys()]
        with open(filename, "r") as f:
            for l in f:
                words = l.split()
                w_ids = self.sent_to_ids(
                    l.strip().split(), trg=True, add_pad=False)
                for i in range(min(k, len(words)-1)):
                    self.kbest[w_ids[0]].append(w_ids[i + 1])
        self.kbest = np.asarray(self.kbest)


def save_vocab(filename, vocab):
    print("Saving lexicon to file %s[.npz]" % filename)
    with open(filename, "wb+") as f:
        pickle.dump(vocab, f)


def load_vocab(filename):
    print("Reading lexicon from file %s[.npz]" % filename)
    with open(filename, "rb") as f:
        vocab = pickle.load(f)
    return vocab
