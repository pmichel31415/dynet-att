# rnn LM

from __future__ import print_function, division
import numpy as np
import argparse
from collections import defaultdict
import dynet as dy
import time
import pickle
import sys

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
parser.add_argument("--test_src", '-tes',
                    default='en-de/test.en-de.en', type=str)
parser.add_argument("--test_out", '-teo',
                    default='results/test.en-de.de', type=str)
parser.add_argument("--model", '-m', type=str, help='Model to load from')
parser.add_argument('--num_epochs', '-ne',
                    type=int, help='Number of epochs', default=1)
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
parser.add_argument('--test_every', '-te',
                    type=int, help='Run on test set every', default=500)
parser.add_argument('--max_len', '-ml', type=int,
                    help='Maximum length of generated sentences', default=60)
parser.add_argument("--attention", '-att',
                    help="Use attention",
                    action="store_true")
parser.add_argument("--exp_name", '-en', type=str, required=True,
                    help='Name of the experiment (used so save the model)')
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


def reverse_dic(dic):
    rev_dic = dict()
    for k, v in dic.items():
        rev_dic[v] = k
    return rev_dic


def read_corpus(file, dic, frozen=False):
    # for each line in the file, split the words and turn them into IDs like
    # this:
    sentences = []
    with open(file, 'r') as f:
        for l in f:
            sent = [dic['SOS']]
            if frozen:
                for w in l.split():
                    if w not in dic:
                        sent.append(dic['UNK'])
                    else:
                        sent.append(dic[w])
            else:
                sent += [dic[w] for w in l.split()]
            sent.append(dic['EOS'])
            sentences.append(sent)
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
            raise StopIteration()
        idxs = self.order[self.i:min(self.i + self.bs, self.n)]
        self.i += self.bs
        return self.datas[idxs], self.datat[idxs]

    def __next__(self):
        return self.next()

    def __iter__(self): return self


class Seq2SeqModel(dy.Saveable):

    def __init__(self,
                 model,
                 input_dim,
                 hidden_dim,
                 source_vocab_size,
                 target_vocab_size,
                 attention=False,
                 dropout=0.0,
                 max_len=60):
        # Store config
        self.di, self.dh = input_dim, hidden_dim
        self.vs, self.vt = source_vocab_size, target_vocab_size
        self.att = attention
        self.dr = dropout

        # Declare parameters
        self.enc = dy.VanillaLSTMBuilder(1, self.di, self.dh, model)
        dec_input_dim = self.di + (self.dh if self.att else 0)
        self.dec = dy.VanillaLSTMBuilder(1, dec_input_dim, self.dh, model)
        self.A_p = model.add_parameters((self.dh, self.dh))
        self.MS_p = model.add_lookup_parameters((self.vs, self.di))
        self.MT_p = model.add_lookup_parameters((self.vt, self.di))
        self.D_p = model.add_parameters((self.vt, self.dh))

        self.max_len = max_len

    def calculate_loss(self, src, trg):
        dy.renew_cg()

        bsize = len(src)

        input_len = max(len(s) for s in src)
        output_len = max(len(s) for s in trg)
        # Pad
        if debug:
            start = time.time()
        x = np.zeros((input_len, bsize), dtype=int)
        masksx = np.ones((input_len, self.di, bsize), dtype=float)
        for i in range(bsize):
            srci = src[i][:]
            masksx[:-len(srci), i] = 0.0
            while len(srci) < input_len:
                srci.insert(0, widss['SOS'])
            x[:, i] = srci
        y = np.zeros((output_len, bsize), dtype=int)

        masksy = np.ones((output_len, bsize), dtype=float)
        for i in range(bsize):
            trgi = trg[i][:]
            masksy[len(trgi):, i] = 0.0
            while len(trgi) < output_len:
                trgi.append(widst['EOS'])
            y[:, i] = trgi

        D = dy.parameter(self.D_p)
        A = dy.parameter(self.A_p)

        # Set dropout if necessary
        self.enc.set_dropout(self.dr)
        self.dec.set_dropout(self.dr)
        es = self.enc.initial_state()
        ds = self.dec.initial_state()
        errs = []
        encoded_states = []
        # Encode
        if debug:
            elapsed = time.time()-start
            print('Preprocessing took : ', elapsed)
            start = time.time()
        for iw, mask in zip(x, masksx):
            embs = dy.lookup_batch(self.MS_p, iw)
            es = es.add_input(embs)
            masksx_e = dy.reshape(
                dy.inputMatrix(mask.flatten(order='F'), (self.di, bsize)), (self.di,), batch_size=bsize)
            encoded_states.append(dy.cmult(masksx_e, es.output()))

        if debug:
            elapsed = time.time()-start
            print('Building encoding : ', elapsed)
            start = time.time()
        # Attend
        if self.att:
            H = dy.transpose(dy.concatenate_cols(encoded_states)) * A
        if debug:
            elapsed = time.time()-start
            print('Building attention : ', elapsed)
            start = time.time()
        # Decode
        for cw, nw, mask in zip(y, y[1:], masksy[1:]):
            embs = dy.lookup_batch(self.MT_p, cw)
            if self.att:
                if ds.output() is not None:
                    h = ds.output()
                else:
                    h = dy.zeroes((self.dh,))
                context = dy.transpose(H) * dy.softmax(H * h)
                ds = ds.add_input(dy.concatenate([embs, context]))
            else:
                ds = ds.add_input(embs)
            s = D * ds.output()
            masksy_e = dy.reshape(dy.inputVector(mask), (1,), batch_size=bsize)
            err = dy.cmult(dy.pickneglogsoftmax_batch(s, nw), masksy_e)
            errs.append(err)
            # print(cw,nw,mask)
        if debug:
            elapsed = time.time()-start
            print('Building decoding : ', elapsed)
        err = dy.sum_batches(dy.esum(errs))

        return err

    def translate(self, x, decoding='greedy', T=1.0):
        dy.renew_cg()
        input_len = len(x)
        D = dy.parameter(self.D_p)
        A = dy.parameter(self.A_p)
        self.enc.disable_dropout()
        self.dec.disable_dropout()
        es = self.enc.initial_state()
        ds = self.dec.initial_state()
        encoded_states = []
        # Encode
        if debug:
            start = time.time()
        for w in x:
            embs = dy.lookup(self.MS_p, w)
            es = es.add_input(embs)
            encoded_states.append(es.output())

        if debug:
            elapsed = time.time()-start
            print('Building encoding : ', elapsed)
            start = time.time()
        # Attend
        if self.att:
            H = dy.transpose(dy.concatenate_cols(encoded_states)) * A
        if debug:
            elapsed = time.time()-start
            print('Building attention : ', elapsed)
            start = time.time()
        # Decode
        cw = widst['SOS']
        words = []
        for i in range(int(min(self.max_len, input_len * 1.5))):
            embs = dy.lookup(self.MT_p, cw)
            if self.att:
                if ds.output() is not None:
                    h = ds.output()
                else:
                    h = dy.zeroes((self.dh,))
                context = dy.transpose(H) * dy.softmax(H * h)
                ds = ds.add_input(dy.concatenate([embs, context]))
            else:
                ds = ds.add_input(embs)
            s = D * ds.output()
            p = dy.softmax(s * (1 / T)).npvalue()
            # Careful of float error
            p = p/p.sum()
            if decoding == 'sample':
                nw = np.random.choice(range(len(widst)), p=p)
            elif decoding == 'greedy':
                nw = np.argmax(p)

            words.append(nw)
            if nw == widst['EOS']:
                break
            else:
                cw = nw

        if debug:
            elapsed = time.time()-start
            print('Decoding : ', elapsed)

        return words

    def get_components(self):
        return self.MS_p, self.MT_p, self.D_p, self.enc, self.dec, self.A_p

    def restore_components(self, components):
        self.MS_p, self.MT_p, self.D_p, self.enc, self.dec, self.A_p = components


if __name__ == '__main__':

    # ===================================================================
    if verbose:
        print('Reading corpora')
    trainings_data = read_corpus(args.train_src, widss)
    trainingt_data = read_corpus(args.train_dst, widst)
    valids_data = read_corpus(args.valid_src, widss)
    validt_data = read_corpus(args.valid_dst, widst)
    tests_data = read_corpus(args.test_src, widss, frozen=True)

    ids2ws = reverse_dic(widss)
    ids2wt = reverse_dic(widst)

    # ===================================================================
    if verbose:
        print('Creating model')
        sys.stdout.flush()
    m = dy.Model()
    model_file = args.model
    if model_file is not None:
        [s2s] = m.load(model_file)
    else:
        s2s = Seq2SeqModel(m,
                           args.emb_dim,
                           args.hidden_dim,
                           len(widss),
                           len(widst),
                           attention=args.attention,
                           dropout=args.dropout_rate,
                           max_len=args.max_len)
        model_file = args.exp_name+'_model.txt'


    trainer = dy.SimpleSGDTrainer(m, args.learning_rate, args.learning_rate_decay)

    # ===================================================================
    if verbose:
        print_config()
        sys.stdout.flush()

    # ===================================================================
    if verbose:
        print('Creating batch loaders')
        sys.stdout.flush()
    trainbatchloader = BatchLoader(
        trainings_data, trainingt_data, args.batch_size)
    devbatchloader = BatchLoader(valids_data, validt_data, args.batch_size)

    # ===================================================================
    if verbose:
        print('starting training')
        sys.stdout.flush()
    train_loss = 0
    start = time.time()
    processed = 0

    best_dev_loss = -np.inf
    for epoch in range(args.num_epochs):
        for i, (x, y) in enumerate(trainbatchloader):
            processed += sum(map(len, y))
            loss = s2s.calculate_loss(x, y)
            loss.backward()
            trainer.update()
            train_loss += loss.scalar_value()
            if (i+1) % args.check_train_error_every == 0:
                logloss = train_loss / processed
                ppl = np.exp(logloss)
                elapsed = time.time()-start
                print("[epoch %d] Training_loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
                      (epoch, logloss, ppl, elapsed, processed))
                start = time.time()
                train_loss = 0
                processed = 0
                sys.stdout.flush()
            if (i+1) % args.check_valid_error_every == 0:
                trainer.update_epoch()
                dev_loss = 0
                dev_processed = 0
                dev_start = time.time()
                for x, y in devbatchloader:
                    dev_processed += sum(map(len, y))
                    loss = s2s.calculate_loss(x, y)
                    dev_loss += loss.scalar_value()
                dev_logloss = dev_loss/dev_processed
                dev_ppl = np.exp(dev_logloss)
                dev_elapsed = time.time()-dev_start
                print("[epoch %d] Dev loss=%f, ppl=%f, time=%f s, tokens processed=%d" %
                      (epoch, dev_logloss, dev_ppl, dev_elapsed, dev_processed))
                if dev_ppl > best_dev_loss:
                    print('Best dev error up to date, saving model to', model_file)
                    m.save(model_file, [s2s])
                sys.stdout.flush()

            if (i+1) % args.test_every == 0:
                dev_loss = 0
                j = 0
                print('Start running on test set, buckle up!')
                test_start = time.time()
                with open(args.test_out, 'w+') as of:
                    for x in tests_data:
                        y = s2s.translate(x, decoding='greedy')
                        translation = ' '.join([ids2wt[w] for w in y])
                        source = ' '.join([ids2ws[w] for w in x])
                        of.write(source + ':' + translation+'\n')
                test_elapsed = time.time()-test_start
                print('Finished running on test set,' test_elapsed,'elapsed.')
                sys.stdout.flush()

