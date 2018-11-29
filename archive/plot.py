#!/usr/bin/env python3
import sys
from os.path import isfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

color_train='#64DD17'
color_dev='#FF6F00'
color_bleu='#2979FF'

log_file = open(sys.argv[1],'r')
plot_filename = 'plots/' + sys.argv[1][4:-3] + 'png'

train_ppl = []
valid_ppl = []
valid_bleu = []
for l in log_file:
    if 'exp_name :' in l:
        exp_name = l.strip().split(':')[1][1:]
    if 'check_valid_error_every :' in l:
        valid_interval = int(l.strip().split(':')[1][1:])
    if "Training_loss" in l:
        ppl = float(l.strip().split('ppl=')[-1].split(', time')[0])
        train_ppl.append(ppl)
    if "Dev loss=" in l:
        ppl = float(l.strip().split('ppl=')[-1].split(', time')[0])
        valid_ppl.append(ppl)
    if "BLEU = " in l:
        bleu = float(l.strip().split(', ')[0][7:])
        valid_bleu.append(bleu)

test_bleu = None
if len(valid_bleu) > len(valid_ppl):
    test_bleu = valid_bleu[-1]
    valid_bleu = valid_bleu[:-1]

iterations = np.arange(0, len(valid_bleu), 2)
valid_ticks = [str((5 * (i+1)) ) + 'k' for i in iterations]
train_ticks = [str((5 * (i)) ) + 'k' for i in iterations]

# Plot training ppl
plt.subplot('311')
plt.plot(range(len(train_ppl)), train_ppl,linewidth=1,alpha=0.5,label='Train perplexity (min = %.2f)' % min(train_ppl),color=color_train)
plt.xlabel('Iterations')
plt.ylabel('Perplexity')
plt.xticks(iterations * 25, train_ticks)
plt.legend(loc='upper right')
plt.grid()

# Plot dev ppl
plt.subplot('312')
plt.plot(range(len(valid_ppl)), valid_ppl,linewidth=3,alpha=0.8,label='Validation perplexity (min = %.2f)' % min(valid_ppl),color=color_dev)
plt.xlabel('Iterations')
plt.ylabel('Perplexity')
plt.xticks(iterations,valid_ticks)
plt.legend(loc='upper right')
plt.grid()

# Plot dev BLEU
plt.subplot('313')
plt.plot(range(len(valid_bleu)), valid_bleu,linewidth=3,alpha=0.8,label='Validation BLEU score, with beam size 3 (max = %.2f)' % max(valid_bleu),color=color_bleu)
plt.xlabel('Iterations')
plt.ylabel('BLEU score')
plt.xticks(iterations,valid_ticks)
plt.legend(loc='lower right')
plt.grid()

plt.savefig(plot_filename,dpi=600)
