# rnn LM

from __future__ import print_function, division
import numpy as np
import argparse
from collections import defaultdict
import dynet as dy
import time

CHECK_TRAIN_ERROR_EVERY = 10
CHECK_VALID_ERROR_EVERY = 1000

widss = defaultdict(lambda: len(widss))
widst = defaultdict(lambda: len(widst))

parser = argparse.ArgumentParser()
parser.add_argument("--dynet-seed", default=0, type=int)
parser.add_argument("--dynet-mem", default=512, type=int)
parser.add_argument("--dynet-gpus", default=0, type=int)
parser.add_argument("--train_src", '-ts',
                    default='en-de/train.en-de.en', type=str)
parser.add_argument("--train_dst", '-td',
                    default='en-de/train.en-de.de', type=str)
parser.add_argument("--valid_src", '-vs',
                    default='en-de/valid.en-de.en', type=str)
parser.add_argument("--valid_dst", '-vd',
                    default='en-de/valid.en-de.de', type=str)
parser.add_argument('--batch_size', '-bs',
                    type=int, help='minibatch size', default=20)
parser.add_argument('--emb_dim', '-de',
                    type=int, help='embedding size', default=256)
parser.add_argument('--hidden_dim', '-dh',
                    type=int, help='hidden size', default=256)
parser.add_argument('--dropout_rate', '-dr',
                    type=float, help='dropout rate', default=0.0)
parser.add_argument('--learning_rate', '-lr',
                    type=float, help='learning rate', default=1.0)
parser.add_argument('--learning_rate_decay', '-lrd',
                    type=float, help='learning rate decay', default=0.0)
parser.add_argument('--check_train_error_every', '-ct',
                    type=int, help='Check train error every', default=100)
parser.add_argument('--check_valid_error_every', '-cv',
                    type=int, help='Check valid error every', default=1000)
parser.add_argument("--attention", '-att',
                    help="Use attention",
                    action="store_true")
parser.add_argument("--verbose", '-v',
                    help="increase output verbosity",
                    action="store_true")
parser.add_argument("--debug", '-dbg',
                    help="Print debugging info",
                    action="store_true")
args = parser.parse_args()

verbose = args.verbose
debug = args.debug

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


def print_config():
    print('======= CONFIG =======')
    for k, v in vars(args).items():
        print(k, ':', v)
    print('Source vocabulary size :', len(widss))
    print('Target vocabulary size :', len(widst))
    print('======================')


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


class Seq2SeqModel(object):

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 source_vocab_size,
                 target_vocab_size,
                 lr=1.0,
                 lr_decay=0.0,
                 attention=False,
                 dropout=0.0):
        # Store config
        self.di, self.dh = input_dim, hidden_dim
        self.vs, self.vt = source_vocab_size, target_vocab_size
        self.att = attention
        self.dr = dropout

        self.model = dy.Model()

        # Declare parameters
        self.enc = dy.VanillaLSTMBuilder(1, self.di, self.dh, self.model)
        dec_input_dim = self.di + (self.dh if self.att else 0)
        self.dec = dy.VanillaLSTMBuilder(1, dec_input_dim, self.dh, self.model)
        self.MS_p = self.model.add_lookup_parameters((self.vs, self.di))
        self.MT_p = self.model.add_lookup_parameters((self.vt, self.di))
        self.D_p = self.model.add_parameters((self.vt, self.dh))

        self.trainer = dy.SimpleSGDTrainer(self.model, lr, lr_decay)

    def calculate_loss(self, src, trg, update=False):
        dy.renew_cg()

        bsize = len(src)

        input_len = max(len(s) for s in src)
        output_len = max(len(s) for s in trg)
        # Pad
        if debug:
            start = time.time()
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

        D = dy.parameter(self.D_p)
        es = self.enc.initial_state()
        ds = self.dec.initial_state()
        err = dy.scalarInput(0)
        encoded_states = []
        # Encode
        if debug:
            elapsed = time.time()-start
            print('Preprocessing took : ', elapsed)
            start = time.time()
        for i in range(input_len):
            embs = dy.lookup_batch(self.MS_p, x[i])
            es = es.add_input(embs)
            encoded_states.append(es.output())

        if debug:
            elapsed = time.time()-start
            print('Building encoding : ', elapsed)
            start = time.time()
        # Attend
        if self.att:
            H = dy.transpose(dy.concatenate_cols(encoded_states))
        if debug:
            elapsed = time.time()-start
            print('Building attention : ', elapsed)
            start = time.time()
        # Decode
        for j in range(output_len-1):
            embs = dy.lookup_batch(self.MT_p, y[j])
            if self.att:
                if j > 0:
                    context=dy.transpose(H) * dy.softmax(H * ds.h()[-1])
                else:
                    context=dy.zeroes((self.dh,),batch_size=bsize)
                ds = ds.add_input(dy.concatenate([embs, context]))
            else:
                ds = ds.add_input(embs)
            s = D * ds.output()
            err += dy.pickneglogsoftmax_batch(s, y[j+1])
        if debug:
            elapsed = time.time()-start
            print('Building decoding : ', elapsed)
            start = time.time()
        err = dy.sum_batches(err) * (1 / bsize)
        error = err.scalar_value()
        if debug:
            elapsed = time.time()-start
            print('Actually computing stuff : ', elapsed)
            start = time.time()
        if update:
            err.backward()
            self.trainer.update()
        if debug:
            elapsed = time.time()-start
            print('Backward pass : ', elapsed)

        return error

    def transduce(self,x,):


if __name__ == '__main__':

    # ===================================================================
    if verbose:
        print('Reading corpora')
    trainings_data = read_corpus(args.train_src, widss)
    trainingt_data = read_corpus(args.train_dst, widst)
    valids_data = read_corpus(args.valid_src, widss)
    validt_data = read_corpus(args.valid_dst, widst)

    # ===================================================================
    if verbose:
        print('Creating model')
    s2s = Seq2SeqModel(args.emb_dim,
                       args.hidden_dim,
                       len(widss),
                       len(widst),
                       lr=args.learning_rate,
                       lr_decay=args.learning_rate_decay,
                       attention=args.attention,
                       dropout=args.dropout_rate)

    # ===================================================================
    if verbose:
        print_config()

    # ===================================================================
    if verbose:
        print('Creating batch loaders')
    trainbatchloader = BatchLoader(
        trainings_data, trainingt_data, args.batch_size)
    devbatchloader = BatchLoader(valids_data, validt_data, args.batch_size)

    # ===================================================================
    if verbose:
        print('starting training')
    train_loss = 0
    start = time.time()
    processed = 0
    for i, (x, y) in enumerate(trainbatchloader):
        processed += sum(map(len, y))
        loss = s2s.calculate_loss(x, y, update=True)
        train_loss += loss
        if (i+1) % args.check_train_error_every == 0:
            logloss = train_loss / processed
            ppl = np.exp(logloss)
            elapsed = time.time()-start
            print("Training_loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
                  (logloss, ppl, elapsed, processed))
            start = time.time()
            train_loss = 0
            processed = 0
        if (i+1) % args.check_valid_error_every == 0:
            dev_loss = 0
            j = 0
            for x, y in devbatchloader:
                j += sum(map(len, y))
                dev_loss += s2s.calculate_loss(dev_example)
            print("Dev loss=%f" % (dev_loss/j))
