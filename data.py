from __future__ import print_function, division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pickle
import numpy as np
from collections import defaultdict


def save_dic(file, dic):
    """Save dictionary to file
    
    Converts defaultdict to dict to prevent pickling error
    
    Arguments:
        file (str): File path
        dic (defaultdict): Dictionary to save
    """
    with open(file, 'w+') as f:
        pickle.dump(dict(dic), f)


def load_dic(file):
    """Load dictionary from file
    
    Arguments:
        file (str): File path
    
    Returns:
        defaultdict: Loaded dictionary
    """
    with open(file, 'r') as f:
        saved_dic = pickle.load(f)
    dic = defaultdict(lambda: 0)
    for k, v in saved_dic.items():
        dic[k] = v
    return dic, reverse_dic(dic)


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


def read_dic(file, max_size=20000, min_freq=1):
    """Read dictionary from corpus
    
    Args:
        file (str): [description]

    Keyword Arguments:
        max_size (int): Only the top max_size words (by frequency) are stored, the rest is UNKed (default: {20000})
        min_freq (int): Disregard words with frequency <= min_freq (default: {1})
    
    Returns:
        Dictionary
        defaultdict
    """
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
    """Read corpus in list of sentences
    
    Each sentence is a list of integers (determined by dic)
    
    Args:
        file (str): Corpus file path
        dic (defaultdict): Dictionary for the str -> int conversion
    
    Returns:
        Corpus
        list
    """
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
    """Iterator used to load batches
    
    Batches are predetermined so that each batch has only source sentence of the same length (easier for minibatching)
    """

    def __init__(self, datas, datat, bsize):
        """Constructor
        
        Args:
            datas (list): Source corpus
            datat (list): Target corpus
            bsize (int): Batch size
        """
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
        """Reshuffle the batches

        """
        print('Reseeding the dataset')
        self.i = 0
        np.random.shuffle(self.batches)

    def next(self):
        """Get next batch
        
        Returns:
            (source batch, target batch)
            tuple
        
        Raises:
            StopIteration: When all batches have been seen. Also resshuffles the batches
        """
        if self.i >= self.n - 1:
            self.reseed()
            raise StopIteration()
        self.i += 1
        return self.batches[self.i]

    def __next__(self):
        """Same as self.next
        """
        return self.next()

    def __iter__(self): return self
