from __future__ import print_function, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pickle
import numpy as np
from collections import defaultdict


def save_dic(file, dic):
    with open(file, 'w+') as f:
        pickle.dump(dict(dic), f)


def load_dic(file):
    with open(file, 'r') as f:
        saved_dic = pickle.load(f)
    dic = defaultdict(lambda: 0)
    for k, v in saved_dic.items():
        dic[k] = v
    return dic, reverse_dic(dic)


def reverse_dic(dic):
    rev_dic = dict()
    for k, v in dic.items():
        rev_dic[v] = k
    return rev_dic


def read_dic(file, max_size=20000, min_freq=1):
    dic = defaultdict(lambda: 0)
    freqs = defaultdict(lambda: 0)
    dic['UNK'], dic['SOS'], dic['EOS'] = 0, 1, 2
    with open(file, 'r') as f:
        for l in f:
            sent = l.strip().split()
            for word in sent:
                freqs[word] += 1

    sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    for i in range(max_size):
        word, freq = sorted_words[i]
        if freq <= min_freq:
            continue
        dic[word] = len(dic)

    return dic, reverse_dic(dic)


def read_corpus(file, dic):
    # for each line in the file, split the words and turn them into IDs like
    # this:
    sentences = []
    with open(file, 'r') as f:
        for l in f:
            sent = [dic['SOS']]
            for w in l.split():
                if w not in dic:
                    sent.append(dic['UNK'])
                else:
                    sent.append(dic[w])
            sent.append(dic['EOS'])
            sentences.append(sent)
    return sentences


class BatchLoader(object):

    def __init__(self, datas, datat, bsize):
        self.batches = []

        self.bs = bsize

        # Bucket samples by source sentence length
        buckets = defaultdict(list)
        for src, trg in zip(datas, datat):
            buckets[len(src)].append((src, trg))

        for src_len, bucket in buckets.items():
            np.random.shuffle(bucket)
            num_batches = int(np.ceil(len(bucket) * 1.0 / self.bs))
            for i in range(num_batches):
                cur_batch_size = self.bs if i < num_batches - 1 else len(bucket) - self.bs * i
                self.batches.append(([bucket[i * self.bs + j][0] for j in range(cur_batch_size)],
                                     [bucket[i * self.bs + j][1] for j in range(cur_batch_size)]))

        self.n = len(self.batches)
        self.reseed()

    def reseed(self):
        print('Reseeding the dataset')
        self.i = 0
        np.random.shuffle(self.batches)

    def next(self):
        if self.i >= self.n - 1:
            self.reseed()
            raise StopIteration()
        self.i += 1
        return self.batches[self.i]

    def __next__(self):
        return self.next()

    def __iter__(self): return self
