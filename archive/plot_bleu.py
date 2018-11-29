#!/usr/bin/env python3
import sys
from os.path import isfile
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["font.size"] = "13"


def get_bleus(filename):
    bleus = []
    with open(filename, 'r') as f:
        for l in f:
            match = re.match('BLEU = ([\.0-9]*),.*', l)
            if match:
                bleus.append(match.group(1))
            end_of_training = re.match('No improvement.*', l)
            if end_of_training:
                break
        return np.asarray(bleus, dtype=float)

def adapt_ylim(values):
    ymin, ymax = plt.ylim()
    ymin = min(ymin, values.min() - 0.5)
    ymax = max(ymax, values.max() + 0.5)
    plt.ylim([ymin, ymax])

def plot_bleus(filename, name, color='blue', marker='.', N=10):
    x = get_bleus(filename)
    #test = x[-1]
    adapt = x#[:-1]
    n = len(adapt)
    #plt.plot([1, n], [test, test], '--', color=color)
    plt.plot(range(1, n+1), adapt, color=color, label=name, linewidth=4,alpha=.8, marker=marker)
    adapt_ylim(x)

colors= ['orangered', 'royalblue', 'limegreen']
markers = ['o', 'x', '+']

for i, ls in enumerate([0.0, 0.1, 0.5]):
    plot_bleus('backup/log_iwslt_de_en_%.1f_syn_smoothing.txt' % ls, '$\\varepsilon = %.1f$' % ls, colors[i], markers[i])

plt.xlabel('Epochs')
plt.ylabel('BLEU')

plt.title('Effect of label smoothing coefficient')
plt.legend()

plt.savefig('iwslt_label_smoothing.png', size=(12,8), dpi=200)
plt.show()
