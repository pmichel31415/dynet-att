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

max_beam=10

def get_bleus(filename):
    bleus = []
    with open(filename, 'r') as f:
        for l in f:
            match = re.match('BLEU = ([\.0-9]*),.*', l)
            if match:
                bleus.append(match.group(1))
        return np.asarray(bleus, dtype=float)

def adapt_ylim(values):
    ymin, ymax = plt.ylim()
    ymin = min(ymin, values.min() - 0.5)
    ymax = max(ymax, values.max() + 0.5)
    plt.ylim([ymin, ymax])

def plot_beam(filename, name, color='blue', marker='.', N=10):
    x = get_bleus(filename)
    bleus = x[-max_beam-1:-1]
    n = len(bleus)
    plt.plot(range(1, n+1), bleus, color=color, label=name, linewidth=4,alpha=.8, marker=marker)
    adapt_ylim(bleus)

colors= ['orangered', 'royalblue', 'limegreen']
markers = ['o', 'x', '+']

for i, ls in enumerate([0.0, 0.1, 0.5]):
    plot_beam('output/log_iwslt_de_en_%.1f_ppl.txt' % ls, '$\\varepsilon = %.1f$' % ls, colors[i], markers[i])

plt.xlabel('Beam size')
plt.ylabel('BLEU')

plt.title('Effect of beam size for different label smoothing values')
plt.legend()

plt.savefig('iwslt_label_smoothing_beam.png', size=(12,8), dpi=200)
plt.show()
