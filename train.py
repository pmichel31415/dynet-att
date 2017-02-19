# rnn LM

from __future__ import print_function, division
import numpy as np
from collections import defaultdict
import dynet as dy
import time

CHECK_TRAIN_ERROR_EVERY = 10
CHECK_VALID_ERROR_EVERY = 1000

attention = False

widss = defaultdict(lambda: len(widss))
widst = defaultdict(lambda: len(widst))
data = []
trains_file = 'en-de/train.en-de.en'
traint_file = 'en-de/train.en-de.de'
valids_file = 'en-de/valid.en-de.en'
validt_file = 'en-de/valid.en-de.de'

# Writing a function to read in the training and test corpora, and
# converting the words into numerical IDs.


def read_corpus(file, dic):
    # for each line in the file, split the words and turn them into IDs like
    # this:
    sentences = []
    with open(file, 'r') as f:
        for l in f:
            sentences.append([dic['SOS']]+[dic[w]
                                           for w in l.split()]+[dic['EOS']])
    return sentences

print('Reading corpora')
trainings_data = read_corpus(trains_file, widss)
trainingt_data = read_corpus(traint_file, widst)
valids_data = read_corpus(valids_file, widss)
validt_data = read_corpus(validt_file, widst)

print('Creating model')
VOCAB_SIZE_S = len(widss)
VOCAB_SIZE_T = len(widst)
EMB_DIM = 128
HID_DIM = 128
BATCH_SIZE = 20

model = dy.Model()

enc = dy.VanillaLSTMBuilder(1, EMB_DIM, HID_DIM, model)
dec = dy.VanillaLSTMBuilder(
    1, HID_DIM * (2 if attention else 1), HID_DIM, model)
MS_p = model.add_lookup_parameters((VOCAB_SIZE_S, EMB_DIM))
MT_p = model.add_lookup_parameters((VOCAB_SIZE_T, EMB_DIM))
D_p = model.add_parameters((VOCAB_SIZE_T, HID_DIM))

trainer = dy.SimpleSGDTrainer(model, 1.0, 0.01)


class BatchLoader(object):

    def __init__(self, datas, datat, bsize):
        self.datas = np.asarray(datas, dtype=list)
        self.datat = np.asarray(datat, dtype=list)
        self.n = len(self.datas)
        self.bs = bsize
        self.order = np.arange(self.n, dtype=int)
        self.reseed()

    def reseed(self):
        print('Reseeding the dataset')
        self.i = 0
        np.random.shuffle(self.order)

    def next(self):
        if self.i >= self.n:
            self.reseed()
        idxs = self.order[self.i:min(self.i + self.bs, self.n)]
        self.i += self.bs
        return self.datas[idxs], self.datat[idxs]

    def __next__(self):
        return self.next()

    def __iter__(self): return self


def calculate_loss(src, trg, update=False):
    dy.renew_cg()
    # M = dy.parameter(M_p)
    # print(x.shape)

    bsize = len(src)

    input_len = max(len(s) for s in src)
    output_len = max(len(s) for s in trg)
    # Pad
    x = np.zeros((input_len, bsize), dtype=int)
    for i in range(bsize):
        while len(src[i]) < input_len:
            src[i].insert(0, widss['SOS'])
        x[:, i] = src[i]
    y = np.zeros((output_len, bsize), dtype=int)
    for i in range(bsize):
        while len(trg[i]) < output_len:
            trg[i].append(widst['EOS'])
        y[:, i] = trg[i]

    print(input_len)
    print(output_len)
    D = dy.parameter(D_p)
    es = enc.initial_state()
    ds = dec.initial_state()
    err = dy.scalarInput(0)
    encoded_states = []
    # Encode
    for i in range(input_len):
        embs = dy.lookup_batch(MS_p, x[i])
        es = es.add_input(embs)
        encoded_states.append(es.output())
    # Attend
    if attention:
        H = dy.transpose(dy.concatenate_cols(encoded_states))
    # Decode
    for j in range(output_len-1):
        embs = dy.lookup_batch(MT_p, y[j])
        if attention:
            context = dy.softmax(H * ds.h()[-1]) * H if j > 0 else dy.zeroes()
            ds = ds.add_input(dy.concatenate(embs, context))
        else:
            ds = ds.add_input(embs)
        s = D * ds.output()
        err += dy.pickneglogsoftmax_batch(s, y[j+1])
    err = dy.sum_batches(err) * (1 / BATCH_SIZE)
    error = err.value()
    if update:
        err.backward()
        trainer.update()

    return error,output_len * bsize

print('Source vocabulary size :',len(widss))
print('Target vocabulary size :',len(widst))

print('Creating batch loaders')
trainbatchloader = BatchLoader(trainings_data, trainingt_data, BATCH_SIZE)
devbatchloader = BatchLoader(valids_data, validt_data, BATCH_SIZE)
train_loss = 0
print('starting training')
start = time.time()
processed = 0
for i, (x, y) in enumerate(trainbatchloader):
    loss,ntokens = calculate_loss(x, y, update=True)
    train_loss += loss
    processed += ntokens
    if (i+1) % CHECK_TRAIN_ERROR_EVERY == 0:
        logloss = train_loss / processed
        ppl = np.exp(logloss)
        elapsed = time.time()-start
        print("Training_loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
              (logloss, ppl, elapsed, processed))
        start = time.time()
        train_loss = 0
        processed=0
    if (i+1) % CHECK_VALID_ERROR_EVERY == 0:
        dev_loss = 0
        j = 0
        for x,y in devbatchloader:
            j += sum(map(len,y))
            dev_loss += calculate_loss(dev_example)
        print("Dev loss=%f" % (dev_loss/j))
