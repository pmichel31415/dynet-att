from __future__ import division

import sys
import numpy as np
import utils
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def frequencies(filename):
    dic = defaultdict(lambda: 0)
    with open(filename, 'r') as f:
        for l in f:
            for w in l.strip().split():
                dic[w]+=1
    freqs = np.asarray(dic.values(), dtype=float)
    freqs /= freqs.sum()
    return np.sort(freqs)[::-1]

if __name__ == '__main__':
    max_rank=10000
    for filename in sys.argv[1:]:
        freqs = frequencies(filename)
        #plt.hist(freqs[:max_rank], bins=int(np.sqrt(max_rank)), normed=True, alpha=0.5, label=filename)
        plt.loglog(freqs[:max_rank], label=filename)
    plt.legend()
    plt.savefig('zipf.png', size=(12,12), dpi=200)

