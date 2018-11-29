import sys
import utils
from collections import defaultdict
import numpy as np

train = sys.argv[1]
ref_file = sys.argv[2]
out_file = sys.argv[3]
res_file = sys.argv[4]

def read_dic(file, max_size=200000, min_freq=-1):
    dic = defaultdict(lambda: 0)
    freqs = defaultdict(lambda:0)
    dic['UNK'], dic['SOS'], dic['EOS'] = 0, 1, 2
    with open(file, 'r') as f:
        for l in f:
            sent = l.strip().split()
            for word in sent:
                freqs[word] += 1
    sorted_words = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    freqs = [0, 0, 0]
    for i in range(min(max_size, len(sorted_words))):
        word, freq = sorted_words[i]
        if freq < min_freq:
            continue
        dic[word] = len(dic)
        freqs.append(freq)
    freqs = np.asarray(freqs)
    freqs = freqs / freqs.sum()

    return dic, freqs


# Dictionary and frequencies
dic, freqs = read_dic(train, max_size=40000)

# Gold translation
ref = utils.loadtxt(ref_file)
# Output of the system
out = utils.loadtxt(out_file)

# Number of times the word appears in the reference
num_ref = np.zeros(len(dic))
# Number of times the word appears in the output
num_out = np.zeros(len(dic))
# Number of times the word appears in both
num_both = np.zeros(len(dic))

for y, y_ in zip(ref, out):
    s, s_ = y.split(), y_.split()
    for w in s:
        num_ref[dic[w]] += 1
        if w in s_:
            num_both[dic[w]] += 1
    for w_ in s_:
        num_out[dic[w_]] += 1

# Precision
P = num_both / (num_out + 1e-20)
# Recall
R = num_both / (num_ref + 1e-20)
# F-score
F = 2 * P * R / (P + R+ 1e-20)

# Sort by decreasing frequency
ids = np.argsort(freqs)[::-1]
F = F[ids]
P = P[ids]
R = R[ids]
num_ref = num_ref[ids]

#print(P,R,F)
bin_size=2000
bin_vals = np.zeros((3, len(F) // bin_size))
for i in range(0, len(F)-bin_size, bin_size):
    nonzeros = num_ref[i:i + bin_size] != 0
    avg_p = P[i:i + bin_size][nonzeros].mean()
    avg_r = R[i:i + bin_size][nonzeros].mean()
    avg_f = F[i:i + bin_size][nonzeros].mean()
    bin_vals[:, i//bin_size] = [avg_p, avg_r, avg_f]

np.savetxt(res_file, bin_vals)

# Bin according to frequency and only retain words that appear in the reference
top_500 = F[:500][num_ref[:500]!=0]
top_500_2000 = F[500:2000][num_ref[500:2000]!=0]
top_2000_10000 = F[2000:10000][num_ref[2000:10000]!=0]
top_10000_40000 = F[10000:40000][num_ref[10000:40000]!=0]

# Print percentages
print('F measure for file %s' % out_file)
print('1-500: %.2f%%' % (top_500.mean() * 100))
print('500-2000: %.2f%%' % (top_500_2000.mean() * 100))
print('2000-10000: %.2f%%' % (top_2000_10000.mean() * 100))
print('10000-40000: %.2f%%' % (top_10000_40000.mean() * 100))
print('Total: %.2f%%' % (F[num_ref!=0].mean() * 100))



